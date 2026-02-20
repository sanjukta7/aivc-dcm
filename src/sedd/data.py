import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union, List
import warnings

Tensor = torch.Tensor
Array = np.ndarray


class RNASeqDataset(Dataset):

    def __init__(
        self,
        expression: Union[Tensor, Array],
        gene_names: Optional[List[str]] = None,
        cell_labels: Optional[Union[Tensor, Array]] = None,
        num_bins: int = 100,
    ):
        self.expression = expression if isinstance(expression, Tensor) else torch.from_numpy(expression).long()
        self.metadata = {"num_bins": num_bins, "method": "precomputed"}

        self.num_cells, self.num_genes = self.expression.shape
        self.num_bins = num_bins
        self.gene_names = gene_names

        if cell_labels is not None:
            if isinstance(cell_labels, np.ndarray):
                cell_labels = torch.from_numpy(cell_labels)
            self.cell_labels = cell_labels
        else:
            self.cell_labels = None

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> Tensor:
        """Get a single cell's expression profile."""
        return self.expression[idx]

    def get_with_label(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Get expression with cell label."""
        expr = self.expression[idx]
        label = self.cell_labels[idx] if self.cell_labels is not None else None
        return expr, label

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


def train_val_split(
    dataset: RNASeqDataset,
    val_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:

    from torch.utils.data import Subset

    num_samples = len(dataset)
    num_val = int(num_samples * val_fraction)
    num_train = num_samples - num_val

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


class PerturbSeqDataset(Dataset):
    """Dataset for perturbation-seq data with control-perturbed pairs.

    This dataset handles single-cell perturbation data where each sample consists of:
    - Control cell expression (baseline/unperturbed)
    - Perturbation label (which gene/condition was perturbed)
    - Perturbed cell expression (outcome after perturbation)

    The dataset expects AnnData format with:
    - X: Expression matrix (cells × genes), already discretized
    - obs[pert_col]: Column containing perturbation labels
    - obs[control_col]: Optional column marking control cells

    For training, control cells are matched with perturbed cells having the same
    perturbation label to create (control, perturbation, perturbed) triplets.
    """

    def __init__(
        self,
        expression: Union[Tensor, Array],
        pert_labels: Union[Tensor, Array, List[str]],
        control_expression: Optional[Union[Tensor, Array]] = None,
        gene_names: Optional[List[str]] = None,
        num_bins: int = 100,
        control_pert_name: str = "control",
    ):
        """
        Args:
            expression: Expression matrix [num_cells, num_genes], already discretized
            pert_labels: Perturbation labels [num_cells], either indices or names
            control_expression: Optional separate control expression matrix.
                If None, control cells are identified from pert_labels.
            gene_names: Optional list of gene names
            num_bins: Number of expression bins
            control_pert_name: Name of control perturbation in pert_labels
        """
        # Convert to tensors
        if not isinstance(expression, Tensor):
            expression = torch.from_numpy(expression).long()
        self.expression = expression

        self.num_cells, self.num_genes = expression.shape
        self.num_bins = num_bins
        self.gene_names = gene_names
        self.control_pert_name = control_pert_name

        # Handle perturbation labels - convert pandas Categorical to strings if needed
        if hasattr(pert_labels, 'dtype') and 'category' in str(pert_labels.dtype):
            # Numpy array with categorical dtype - convert to strings
            pert_labels = pert_labels.astype(str)
        
        if isinstance(pert_labels, (list, np.ndarray)) and not isinstance(pert_labels[0], (int, np.integer)):
            # String labels - need to encode
            unique_perts = sorted(set(pert_labels))
            self.pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
            self.idx_to_pert = {i: p for p, i in self.pert_to_idx.items()}
            pert_indices = [self.pert_to_idx[p] for p in pert_labels]
            self.pert_labels = torch.tensor(pert_indices, dtype=torch.long)
        else:
            # Already indices
            if not isinstance(pert_labels, Tensor):
                pert_labels = torch.from_numpy(pert_labels).long()
            self.pert_labels = pert_labels
            self.pert_to_idx = None
            self.idx_to_pert = None

        self.num_perturbations = len(torch.unique(self.pert_labels))

        # Handle control expression
        if control_expression is not None:
            if not isinstance(control_expression, Tensor):
                control_expression = torch.from_numpy(control_expression).long()
            self.control_expression = control_expression
            self.has_separate_controls = True
        else:
            # Extract control cells from the main expression matrix
            if self.pert_to_idx is not None:
                control_idx = self.pert_to_idx.get(control_pert_name)
                print(f"DEBUG: Looking for control '{control_pert_name}', got index: {control_idx}")
                print(f"DEBUG: Available perturbations: {list(self.pert_to_idx.keys())[:10]}")
            else:
                control_idx = 0  # Assume 0 is control

            if control_idx is None:
                # Control name not found - use all data as perturbed, sample from all for controls
                print(f"WARNING: Control '{control_pert_name}' not found. Using all data as perturbed cells.")
                self.control_expression = expression
                self.has_separate_controls = False
                # Keep all data as perturbed
                self.expression = expression
                self.pert_labels = self.pert_labels
                self.num_cells = self.expression.shape[0]
            else:
                control_mask = (self.pert_labels == control_idx)
                num_controls = control_mask.sum().item()
                print(f"DEBUG: Found {num_controls} control cells out of {len(self.pert_labels)} total")
                
                self.control_expression = expression[control_mask]
                self.has_separate_controls = False

                # Remove control cells from main dataset
                perturbed_mask = ~control_mask
                self.expression = expression[perturbed_mask]
                self.pert_labels = self.pert_labels[perturbed_mask]
                self.num_cells = self.expression.shape[0]
                print(f"DEBUG: After filtering, {self.num_cells} perturbed cells remaining")

        if len(self.control_expression) == 0:
            warnings.warn(
                f"No control cells found with label '{control_pert_name}'. "
                "Using random sampling from all cells."
            )
            self.control_expression = expression

        print(f"PerturbSeqDataset: {self.num_cells} perturbed cells, "
              f"{len(self.control_expression)} control cells, "
              f"{self.num_perturbations} perturbations")

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a training sample.

        Returns:
            control: Random control cell expression [num_genes]
            pert_label: Perturbation label [scalar]
            perturbed: Perturbed cell expression [num_genes]
        """
        # Get perturbed cell and its label
        perturbed = self.expression[idx]
        pert_label = self.pert_labels[idx]

        # Sample a random control cell
        control_idx = torch.randint(0, len(self.control_expression), (1,)).item()
        control = self.control_expression[control_idx]

        return control, pert_label, perturbed

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )