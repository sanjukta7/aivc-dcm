#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import torch
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader
from cell_load.data_modules import PerturbationDataModule
import scanpy as sc
import os
import yaml

import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import (
    SEDDPerturbationTransformerSmall,
)
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset, train_val_split


# Model registry for easy instantiation by name
MODEL_REGISTRY = {
    "SEDDPerturbationTransformerSmall": SEDDPerturbationTransformerSmall,
}


def find_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a checkpoint directory.

    Priority:
    1. Most recent epoch checkpoint (epoch_*.pt)
    2. final.pt (if no epoch checkpoints exist)
    3. best.pt (if no final checkpoint exists)

    Args:
        checkpoint_dir: Path to directory containing checkpoints

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find most recent epoch checkpoint (highest epoch number)
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if checkpoints:
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    # Check for final checkpoint
    final_ckpt = checkpoint_dir / "final.pt"
    if final_ckpt.exists():
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt

    # Check for best checkpoint
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        print(f"Found best checkpoint: {best_ckpt}")
        return best_ckpt

    return None


def load_yaml_config(config_path):
    config_file = Path(config_path)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def get_config_path(config_loader_path):
    return config_loader_path

def load_conditional_labels(pt_path, pert_names):
    """Load conditional labels from .pt file and create lookup mapping.

    Args:
        pt_path: Path to .pt file containing {pert_name: label} mapping
        pert_names: List of perturbation names in order (for index mapping)

    Returns:
        label_lookup: Tensor of shape [num_perturbations] mapping indices to labels
        missing_perts: List of perturbations not found in .pt file
        total_label_space: int, the total number of labels in the .pt file
                           (i.e. max_label + 1 for scalar labels, or len(pt_data))
    """
    if pt_path is None:
        return None, [], 0

    print(f"\nLoading conditional labels from: {pt_path}")
    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(pt_data, dict):
        raise TypeError(f"Expected .pt file to contain a dictionary, got {type(pt_data)}")

    pt_keys = set(pt_data.keys())
    print(f"Loaded {len(pt_keys)} perturbations from .pt file")

    # Check coverage
    present = [p for p in pert_names if p in pt_keys]
    missing = [p for p in pert_names if p not in pt_keys]

    print(f"Total perturbations in dataset: {len(pert_names)}")
    print(f"Present in .pt file: {len(present)}")
    print(f"Missing from .pt file: {len(missing)}")

    if missing:
        print(f"WARNING: Missing perturbations (first 10): {missing[:10]}")

    # Create lookup tensor: index -> conditional label
    # First, determine the embedding dimension from available data
    embedding_dim = None
    for pert_name in pert_names:
        if pert_name in pt_data:
            label_val = pt_data[pert_name]
            if isinstance(label_val, torch.Tensor):
                if label_val.dim() > 0:  # Vector embedding
                    embedding_dim = label_val.shape[0]
                    break
            else:
                # Scalar labels
                embedding_dim = None
                break
    
    # Compute the total label space from the .pt file
    # For scalar labels: max_label + 1; for vector labels: number of entries
    total_label_space = len(pt_data)
    if embedding_dim is None:
        # Scalar labels — the label space is determined by the max value
        all_vals = []
        for v in pt_data.values():
            if isinstance(v, torch.Tensor):
                all_vals.append(v.item())
            else:
                all_vals.append(int(v))
        if all_vals:
            total_label_space = max(all_vals) + 1
    print(f"Total label space from .pt file: {total_label_space}")

    # Now create the lookup list with proper handling of missing values
    label_lookup = []
    for pert_name in pert_names:
        if pert_name in pt_data:
            label_val = pt_data[pert_name]
            # Handle different formats
            if isinstance(label_val, torch.Tensor):
                label_lookup.append(label_val.cpu())
            else:
                label_lookup.append(torch.tensor(label_val))
        else:
            # Handle missing perturbations based on detected type
            if embedding_dim is not None:
                # Use zero vector for missing embeddings (same shape as others)
                print(f"  Using zero embedding for missing perturbation: {pert_name}")
                label_lookup.append(torch.zeros(embedding_dim))
            else:
                # Use -1 for missing scalar labels
                label_lookup.append(torch.tensor(-1))

    # Stack into tensor
    if len(label_lookup) > 0:
        # Check if labels are scalars or vectors
        if label_lookup[0].dim() == 0:
            # Scalar labels
            label_lookup = torch.stack(label_lookup)
        else:
            # Vector labels (embeddings)
            label_lookup = torch.stack(label_lookup)
    else:
        label_lookup = None

    print(
        f"Created label lookup tensor of shape: {label_lookup.shape if label_lookup is not None else 'None'}"
    )

    return label_lookup, missing, total_label_space

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEDD for perturbation prediction"
    )

    parser.add_argument(
        "--config",
        type=str,
        default= "/home/b5cc/sanjukta.b5cc/st3/configs/perturbseq_small.yaml",
        help="Path to YAML config file (e.g., configs/perturbseq_small.yaml)"
    )

    args, remaining = parser.parse_known_args()
    config = load_yaml_config(args.config)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpointing", {})
    logging_config = config.get("logging", {})
    other_config = config.get("other", {})

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=data_config.get("train_data_path", None),
        help="Path to h5ad file containing perturbation-seq data"
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=data_config.get("gene", "gene"),
        help="Column name in adata.obs containing perturbation labels"
    )
    parser.add_argument(
        "--control_name",
        type=str,
        default=data_config.get("control_name", "control"),
        help="Name of control perturbation in pert_col"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=data_config.get("val_fraction", 0.1),
        help="Fraction of data to use for validation"
    )

    parser.add_argument(
        "--loader_path",
        type=str,
        default=data_config.get("loader_path",None),
        help="this is the toml file for dataloader configs"
    )
    parser.add_argument(
        "--cond_labels_pt_path",
        type=str,
        default=data_config.get("cond_labels_pt_path", None),
        help="Path to .pt file containing conditional labels for perturbations"
    )
    parser.add_argument(
        "--all_perturbations_file",
        type=str,
        default=data_config.get("all_perturbations_file", None),
        help="Path to .txt file listing ALL valid perturbation names (one per line). "
             "Used to set num_perturbations when the .pt file covers more perturbations "
             "than appear in the training data."
    )

    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default=model_config.get("name", "SEDDPerturbationTransformerSmall"),
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use (e.g., SEDDPerturbationTransformerSeparateFiLMSmall)"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=model_config.get("hidden_dim", 128),
        help="Hidden dimension of the transformer"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=model_config.get("num_layers", 4),
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=model_config.get("num_heads", 4),
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=model_config.get("dropout", 0.1),
        help="Dropout rate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=training_config.get("batch_size", 8),
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=training_config.get("num_epochs", 10),
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=training_config.get("learning_rate", 1e-4),
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=training_config.get("weight_decay", 0.01),
        help="Weight decay"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=training_config.get("mask_ratio", 0.15),
        help="Fraction of genes to mask during training"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=training_config.get("gradient_clip", 1.0),
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=training_config.get("use_amp", False),
        help="Use automatic mixed precision training"
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=training_config.get("amp_dtype", "bfloat16"),
        choices=["bfloat16", "float16"],
        help="AMP dtype (bfloat16 or float16)"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=checkpoint_config.get("checkpoint_dir", "experiments/perturbseq_diffusion"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=checkpoint_config.get("save_interval", 2),
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=checkpoint_config.get("resume", None),
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=logging_config.get("log_interval", 5),
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=logging_config.get("val_interval", 1),
        help="Run validation every N epochs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=other_config.get("seed", 42),
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=other_config.get("num_workers", 4),
        help="Number of data loading workers"
    )

    return parser.parse_args()

def print_values(NUM_GENES, NUM_BINS, VOCAB_SIZE):
    print(f"\nData statistics:")
    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading perturbation-seq data from {args.train_data_path}")
    adata = sc.read_h5ad(args.train_data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")
    pert_labels = adata.obs[args.gene].values

    print(f"Found {len(np.unique(pert_labels))} unique perturbations")

    # Extract cell type information
    if 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type'].unique()
        NUM_CELL_TYPES = len(cell_types)
        cell_type_to_idx = {ct: idx for idx, ct in enumerate(sorted(cell_types))}
        print(f"Found {NUM_CELL_TYPES} unique cell types: {sorted(cell_types)}")
    else:
        print("WARNING: 'cell_type' column not found in adata.obs - cell type conditioning disabled")
        NUM_CELL_TYPES = 0
        cell_type_to_idx = {}

    # Compute stats without loading entire dense matrix into memory
    expression_data = adata.X
    NUM_GENES = adata.n_vars
    
    # Compute max value efficiently (works for both sparse and dense)
    if hasattr(expression_data, 'max'):
        # For sparse matrices, .max() returns a matrix, need to get scalar
        max_val = expression_data.max()
        if hasattr(max_val, 'toarray'):
            NUM_BINS = int(max_val.toarray().flatten()[0])
        elif hasattr(max_val, 'item'):
            NUM_BINS = int(max_val.item())
        else:
            NUM_BINS = int(max_val)
    else:
        NUM_BINS = int(np.max(expression_data))
    
    perturbations = adata.obs[args.gene].unique()
    VOCAB_SIZE = NUM_BINS + 1  # +1 for mask token
    NUM_PERTURBATIONS = len(perturbations)

    print_values(NUM_GENES, NUM_BINS, VOCAB_SIZE)
    
    # Compute sparsity efficiently
    if hasattr(expression_data, 'nnz'):
        # Sparse matrix
        total_elements = expression_data.shape[0] * expression_data.shape[1]
        sparsity = 1.0 - (expression_data.nnz / total_elements)
    else:
        sparsity = (expression_data == 0).sum() / expression_data.size
    print(f"Sparsity: {sparsity:.2%}")
    
    # Free the adata object to save memory before DataModule loads it again
    del adata
    import gc
    gc.collect() 
    
    # Create dataset - use perturbations list saved before adata was deleted
    all_genes = perturbations
    train_genes = [g for g in all_genes]

    # Load conditional labels from .pt file if provided
    cond_label_lookup, missing_perts, pt_label_space = load_conditional_labels(
        args.cond_labels_pt_path,
        train_genes
    )

    if cond_label_lookup is not None and len(missing_perts) > 0:
        print(f"WARNING: {len(missing_perts)} perturbations missing from conditional labels file")

    # Override NUM_PERTURBATIONS to cover the full label space from the .pt file
    # or from an explicit all-perturbations file, so that conditional labels
    # (which may index perturbations not in the training set) are always valid.
    if args.all_perturbations_file is not None:
        with open(args.all_perturbations_file, "r") as f:
            all_pert_names = [line.strip() for line in f if line.strip()]
        new_num_perts = len(all_pert_names)
        if new_num_perts != NUM_PERTURBATIONS:
            print(
                f"Overriding NUM_PERTURBATIONS {NUM_PERTURBATIONS} -> "
                f"{new_num_perts} (from all_perturbations_file: {args.all_perturbations_file})"
            )
            NUM_PERTURBATIONS = new_num_perts
    elif pt_label_space > NUM_PERTURBATIONS:
        print(
            f"Overriding NUM_PERTURBATIONS {NUM_PERTURBATIONS} -> "
            f"{pt_label_space} (from .pt file label space)"
        )
        NUM_PERTURBATIONS = pt_label_space

    # Persist derived dimensions for reliable inference (after all overrides)
    args_payload = dict(vars(args))
    args_payload.update(
        {
            "num_genes": NUM_GENES,
            "num_bins": NUM_BINS,
            "num_perturbations": NUM_PERTURBATIONS,
            "vocab_size": VOCAB_SIZE,
            "num_cell_types": NUM_CELL_TYPES,
            "cell_type_to_idx": cell_type_to_idx,
        }
    )
    with open(checkpoint_dir / "args.json", "w") as f:
        json.dump(args_payload, f, indent=2)

    dm = PerturbationDataModule(
        toml_config_path=args.loader_path,
        embed_key=None,  # Use None to read from sparse X matrix directly via fetch_gene_expression
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pert_col=args.gene,
        control_pert=args.control_name,
        perturbations_to_use=train_genes,
        batch_col = "donor",
        cell_sentence_len = 1,
        cell_type_key="cell_type" 
    )

    dm.setup()
    print(f'DataModule setup complete!')

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(type(train_loader))

   
    print(f"Number of training batches: {len(train_loader)}")


    print(type(train_loader))

    # Infer num perturbations from the actual dataloader to avoid label/embedding mismatch.
    # Only override *upward* — never shrink below what the .pt / txt file already set,
    # because the dataloader only sees the training subset while the .pt file covers the
    # full label space (e.g. 91 cytokines vs 81 in training data).
    try:
        first_batch = next(iter(train_loader))
        if isinstance(first_batch, dict):
            pert_emb = first_batch["pert_emb"]
            if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                inferred_num_perts = int(pert_emb.shape[1])
            else:
                inferred_num_perts = int(pert_emb.squeeze(-1).max().item()) + 1
        else:
            # Legacy tuple format: (control, pert_labels, perturbed) or (pert_labels, perturbed)
            if len(first_batch) == 3:
                _, pert_labels, _ = first_batch
            else:
                pert_labels, _ = first_batch
            inferred_num_perts = int(pert_labels.max().item()) + 1

        if inferred_num_perts > NUM_PERTURBATIONS:
            print(
                f"Overriding NUM_PERTURBATIONS {NUM_PERTURBATIONS} -> "
                f"{inferred_num_perts} based on dataloader."
            )
            NUM_PERTURBATIONS = inferred_num_perts
        elif inferred_num_perts < NUM_PERTURBATIONS:
            print(
                f"Dataloader inferred {inferred_num_perts} perturbations, but keeping "
                f"NUM_PERTURBATIONS={NUM_PERTURBATIONS} (set by .pt/txt file)."
            )
    except Exception as exc:
        print(f"WARNING: Could not infer num perturbations from dataloader: {exc}")

    # Infer precomputed embedding dimension from cond_label_lookup if provided
    precomputed_emb_dim = None
    if cond_label_lookup is not None and cond_label_lookup.dim() == 2:
        precomputed_emb_dim = cond_label_lookup.shape[1]
        print(f"Detected precomputed embedding dimension: {precomputed_emb_dim}")

    # Select model class from registry
    model_name = args.model_name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_name]
    print(f"\nCreating {model_name} model...")
    model = ModelClass(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=NUM_GENES,
        precomputed_emb_dim=precomputed_emb_dim,
        num_cell_types=NUM_CELL_TYPES if NUM_CELL_TYPES > 0 else None
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Convert amp_dtype string to torch dtype
    amp_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    amp_dtype = amp_dtype_map.get(args.amp_dtype, torch.bfloat16)
    
    if args.use_amp:
        print(f"\nUsing automatic mixed precision training with dtype: {args.amp_dtype}")

    # Create cell type lookup tensor
    cell_type_lookup = None
    if NUM_CELL_TYPES > 0:
        # Create mapping from perturbation -> cell_type indices for each sample
        # This will be used by the trainer to look up cell types
        cell_type_lookup = cell_type_to_idx
        print(f"Cell type conditioning enabled with {NUM_CELL_TYPES} cell types")
    
    # Create trainer
    trainer = PerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        device=device,
        gradient_clip=args.gradient_clip,
        cond_label_lookup=cond_label_lookup,
        cell_type_lookup=cell_type_lookup,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype
    )

    # Resume from checkpoint if specified
    if args.resume:
        # Support auto-resume from latest checkpoint
        if args.resume.lower() in ["auto", "latest", "last"]:
            print(f"\nAuto-resuming from latest checkpoint in {checkpoint_dir}")
            checkpoint_path = find_checkpoint(checkpoint_dir)
            if checkpoint_path:
                trainer.load_checkpoint(checkpoint_path)
                print(f"Resumed from epoch {trainer.epoch + 1}")
            else:
                print("No checkpoint found, starting from scratch")
        else:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from epoch {trainer.epoch + 1}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        mask_ratio=args.mask_ratio,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Best val loss: {trainer.best_loss:.4f}")

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "final.pt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()