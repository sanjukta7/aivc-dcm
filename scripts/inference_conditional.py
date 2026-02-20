#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import (
    SEDDPerturbationTransformerSmall,
)
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset
from sedd.sampling import PerturbationEulerSampler


# Model registry for easy instantiation by name
MODEL_REGISTRY = {
    "SEDDPerturbationTransformerSmall": SEDDPerturbationTransformerSmall,
}

import yaml
import tomllib


def load_perturbations_from_file(filepath):
    filepath = Path(filepath)
    perturbations = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            perturbations.append(line)

    print(f"Loaded {len(perturbations)} perturbations from {filepath}")
    return perturbations

def load_conditional_labels(pt_path, pert_names):
    """Load conditional labels from .pt file and create lookup mapping.

    Args:
        pt_path: Path to .pt file containing {pert_name: label} mapping
        pert_names: List of perturbation names in order (for index mapping)

    Returns:
        label_lookup: Tensor of shape [num_perturbations] mapping indices to labels
        missing_perts: List of perturbations not found in .pt file
        total_label_space: int, the total number of labels in the .pt file
    """
    if pt_path is None:
        return None, [], 0

    print(f"\nLoading conditional labels from: {pt_path}")
    pt_data = torch.load(pt_path, map_location="cpu")

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

    # Compute the total label space from the .pt file
    total_label_space = len(pt_data)
    if embedding_dim is None:
        all_vals = []
        for v in pt_data.values():
            if isinstance(v, torch.Tensor):
                all_vals.append(v.item())
            else:
                all_vals.append(int(v))
        if all_vals:
            total_label_space = max(all_vals) + 1

    print(f"Created label lookup tensor of shape: {label_lookup.shape if label_lookup is not None else 'None'}")
    print(f"Total label space from .pt file: {total_label_space}")

    return label_lookup, missing, total_label_space

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def find_checkpoint(experiment_dir):
    """
    Find the best or final checkpoint in an experiment directory.

    Priority:
    1. best.pt (best validation checkpoint)
    2. final.pt (final checkpoint)
    3. Most recent epoch checkpoint
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")

    # Check for best checkpoint
    best_ckpt = experiment_dir / "best.pt"
    if best_ckpt.exists():
        print(f"Found best checkpoint: {best_ckpt}")
        return best_ckpt

    # Check for final checkpoint
    final_ckpt = experiment_dir / "final.pt"
    if final_ckpt.exists():
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt

    # Find most recent epoch checkpoint
    checkpoints = list(experiment_dir.glob("epoch_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    raise ValueError(f"No checkpoints found in {experiment_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate perturbed cells from perturbation labels using trained SEDD model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/perturbseq_inference.yaml)"
    )

    args, remaining = parser.parse_known_args()

    config = load_yaml_config(args.config)
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})
    other_config = config.get("other", {})

    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing trained model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file (if not provided, will auto-find best/final)"
    )

    parser.add_argument(
        "--perturbations_file",
        type=str,
        default=inference_config.get("perturbations_file"),
        help="Path to text file with one perturbation label per line"
    )

    parser.add_argument(
        "--perturbations_all_file",
        type=str,
        default=inference_config.get("perturbations_all_file"),
        help="Path to text file with one perturbation label per line"
    )
    
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs='+',
        default=None,
        help="List of perturbation labels to generate (e.g., 'KRAS' 'TP53' 'control')"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=data_config.get("test_data_path"),
        help="Path to h5ad file with test data (for extracting perturbations or evaluation)"
    )
    
    parser.add_argument(
        "--mapping_data_path",
        type=str,
        default=data_config.get("mapping_data_path", data_config.get("train_data_path")),
        help="Path to h5ad file to extract perturbation-to-index mapping (usually training data)"
    )
    parser.add_argument(
        "--cond_labels_pt_path",
        type=str,
        default=data_config.get("cond_labels_pt_path", None),
        help="Path to .pt file containing conditional labels for perturbations"
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=data_config.get("gene", "perturbation"),
        help="Column name in adata.obs containing perturbation labels"
    )
    parser.add_argument(
        "--control_name",
        type=str,
        default=data_config.get("control_name", "control"),
        help="Name of control perturbation"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default=inference_config.get("cell_type", None),
        help="Cell type to condition on (e.g., 'hepg2', 'jurkat', 'rpe1')"
    )
    
    parser.add_argument(
        "--num_samples_per_pert",
        type=int,
        default=inference_config.get("num_samples_per_pert", 10),
        help="Number of cells to generate per perturbation"
    )

    # Inference arguments
    parser.add_argument(
        "--num_steps",
        type=int,
        default=inference_config.get("num_steps", 50),
        help="Number of sampling steps for prediction"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=inference_config.get("temperature", 1.0),
        help="Sampling temperature"
    )

    # Visualization
    parser.add_argument(
        "--num_cells_visualize",
        type=int,
        default=inference_config.get("num_cells_visualize", 5),
        help="Number of individual cells to visualize"
    )
    
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate generated cells against test data (requires --test_data_path)"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=other_config.get("seed", 42),
        help="Random seed"
    )

    return parser.parse_args()

def load_perturbation_labels(args, control_name="non-targeting"):
    """
    1. Loads the master list of perturbations (from .pt mapping if provided).
    2. Assigns indices (Control=0, others=1, 2, 3...).
    3. Returns only the subset requested in args.perturbation_file.
    """
    if args.cond_labels_pt_path:
        pt_data = torch.load(args.cond_labels_pt_path, map_location="cpu")
        if not isinstance(pt_data, dict):
            raise TypeError(
                f"Expected dict in {args.cond_labels_pt_path}, got {type(pt_data)}"
            )
        total_pert_names = list(pt_data.keys())
    else:
        # Load the master list (all perturbations used during training)
        # This list MUST be in the exact same order as used in training
        total_pert_names = load_perturbations_from_file(args.perturbations_all_file)

    # Load the specific subset you want to generate now
    target_pert_names = load_perturbations_from_file(args.perturbations_file)

    # 1 & 2. Recreate the mapping logic: Control is 0, others are +1
    # We remove the control from the loop to ensure it always gets index 0
    others = [p for p in total_pert_names if p != control_name]

    full_mapping = {control_name: 0}
    for i, pert in enumerate(others):
        full_mapping[pert] = i + 1

    # 3. Filter for the requested perturbations and package as list of tuples
    perturbations = []
    for pert in target_pert_names:
        if pert in full_mapping:
            # Packaging as (name, index) to match your generation loop
            perturbations.append((pert, full_mapping[pert]))
        else:
            print(f"Warning: Requested perturbation '{pert}' was not found in the master list.")

    print(f"Loaded {len(perturbations)} perturbations for generation.")
    return perturbations

def print_success(output_dir):
    print("\n" + "="*50)
    print(f"Generation complete! Results saved to {output_dir}")
    print("="*50)
    print(f"\nGenerated files:")
    print(f"  - generated_cells.h5ad (AnnData format)")
    print(f"  - generated_cells.npy (numpy array)")
    print(f"  - perturbation_names.txt (list of perturbations)")
    print(f"  - generation_metadata.json (generation parameters)")
    print(f"  - generation_summary.png (visualizations)")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find checkpoint
    experiment_dir = Path(args.experiment_dir)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint(experiment_dir)

    # Create output directory for results
    output_dir = experiment_dir / "inference_results"
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load training configuration
    args_file = experiment_dir / "args.json"
    if not args_file.exists():
        raise ValueError(f"Training config not found: {args_file}")

    with open(args_file, "r") as f:
        train_config = json.load(f)

    print(f"\nLoaded training configuration from {args_file}")
    print(f"Model: hidden_dim={train_config.get('hidden_dim')}, "
          f"layers={train_config.get('num_layers')}, "
          f"heads={train_config.get('num_heads')}")
          
    NUM_GENES = train_config.get('num_genes')
    NUM_BINS = train_config.get('num_bins') 
    NUM_PERTURBATIONS = train_config.get('num_perturbations')
    
    # Load cell type configuration if available
    NUM_CELL_TYPES = train_config.get('num_cell_types', 0)
    cell_type_to_idx = train_config.get('cell_type_to_idx', {})
    
    if NUM_CELL_TYPES > 0:
        print(f"Model was trained with cell type conditioning: {NUM_CELL_TYPES} cell types")
        print(f"Available cell types: {list(cell_type_to_idx.keys())}")
    else:
        print("Model was trained without cell type conditioning")


    # Prefer checkpoint shapes when available to avoid mismatch
    # Also detect if checkpoint uses precomputed embeddings
    checkpoint_precomputed_dim = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", {})
        if "token_embed.weight" in state_dict:
            ckpt_vocab_size = state_dict["token_embed.weight"].shape[0]
            if NUM_BINS is None or (NUM_BINS + 1) != ckpt_vocab_size:
                NUM_BINS = ckpt_vocab_size - 1
                print(f"Overriding NUM_BINS from checkpoint: {NUM_BINS}")
        if "pert_embed.weight" in state_dict:
            ckpt_num_perts = state_dict["pert_embed.weight"].shape[0]
            if NUM_PERTURBATIONS is None or NUM_PERTURBATIONS != ckpt_num_perts:
                NUM_PERTURBATIONS = ckpt_num_perts
                print(f"Overriding NUM_PERTURBATIONS from checkpoint: {NUM_PERTURBATIONS}")
        if "gene_embed.weight" in state_dict:
            ckpt_num_genes = state_dict["gene_embed.weight"].shape[0]
            if NUM_GENES is None or NUM_GENES != ckpt_num_genes:
                NUM_GENES = ckpt_num_genes
                print(f"Overriding NUM_GENES from checkpoint: {NUM_GENES}")
        # Detect if checkpoint has precomputed projection layer
        if "precomputed_proj.weight" in state_dict:
            checkpoint_precomputed_dim = state_dict["precomputed_proj.weight"].shape[1]
            print(f"Checkpoint was trained with precomputed embeddings (dim={checkpoint_precomputed_dim})")
    except Exception as exc:
        print(f"WARNING: Could not infer dimensions from checkpoint: {exc}")

    VOCAB_SIZE = NUM_BINS + 1

    print(f"\nModel configuration:")
    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Number of perturbations: {NUM_PERTURBATIONS}")


    perturbations = load_perturbation_labels(args, control_name=args.control_name)
    print(f"\nWill generate cells for {len(perturbations)} perturbations:")
    for pert_name, pert_idx in perturbations:
        print(f"  - {pert_name} (index: {pert_idx})")

    # Load conditional labels from .pt file if provided
    # Get all perturbation names from the master list for proper indexing
    total_pert_names = load_perturbations_from_file(args.perturbations_all_file)
    others = [p for p in total_pert_names if p != args.control_name]
    ordered_pert_names = [args.control_name] + others

    cond_label_lookup, missing_perts, pt_label_space = load_conditional_labels(
        args.cond_labels_pt_path,
        ordered_pert_names
    )

    if cond_label_lookup is not None and len(missing_perts) > 0:
        print(f"WARNING: {len(missing_perts)} perturbations missing from conditional labels file")

    # Infer precomputed embedding dimension from checkpoint or cond_label_lookup
    precomputed_emb_dim = None

    # First priority: Use dimension from checkpoint (ensures architecture match)
    if checkpoint_precomputed_dim is not None:
        precomputed_emb_dim = checkpoint_precomputed_dim
        print(f"Using precomputed embedding dimension from checkpoint: {precomputed_emb_dim}")

        # Validate against cond_label_lookup if both are present
        if cond_label_lookup is not None and cond_label_lookup.dim() == 2:
            label_dim = cond_label_lookup.shape[1]
            if label_dim != checkpoint_precomputed_dim:
                raise ValueError(
                    f"Dimension mismatch: checkpoint expects {checkpoint_precomputed_dim} "
                    f"but cond_label_lookup has {label_dim}"
                )
    # Second priority: Infer from cond_label_lookup if provided
    elif cond_label_lookup is not None and cond_label_lookup.dim() == 2:
        precomputed_emb_dim = cond_label_lookup.shape[1]
        print(f"Detected precomputed embedding dimension from labels: {precomputed_emb_dim}")

    # Validate and process cell type argument
    cell_type_idx = None
    if args.cell_type:
        if NUM_CELL_TYPES == 0:
            print(f"\nWARNING: --cell_type '{args.cell_type}' specified but model was not trained with cell type conditioning")
            print("Ignoring cell type argument")
            args.cell_type = None
        elif args.cell_type not in cell_type_to_idx:
            raise ValueError(
                f"Cell type '{args.cell_type}' not found in training data.\n"
                f"Available cell types: {list(cell_type_to_idx.keys())}"
            )
        else:
            cell_type_idx = cell_type_to_idx[args.cell_type]
            print(f"\nGenerating with cell type conditioning: '{args.cell_type}' (index: {cell_type_idx})")
    elif NUM_CELL_TYPES > 0:
        print(f"\nWARNING: Model supports cell type conditioning but --cell_type not specified")
        print(f"Available cell types: {list(cell_type_to_idx.keys())}")
        print("Generating without cell type conditioning (may produce unexpected results)")

    # Create model (use model_name from training config if available)
    model_name = train_config.get("model_name", "SEDDPerturbationTransformerSmall")
    if model_name not in MODEL_REGISTRY:
        print(f"WARNING: Unknown model '{model_name}', falling back to SEDDPerturbationTransformerSmall")
        model_name = "SEDDPerturbationTransformerSmall"

    ModelClass = MODEL_REGISTRY[model_name]
    print(f"\nCreating {model_name} model...")
    model = ModelClass(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        hidden_dim=train_config.get("hidden_dim", 128),
        num_layers=train_config.get("num_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        dropout=train_config.get("dropout", 0.1),
        max_seq_len=NUM_GENES,
        precomputed_emb_dim=precomputed_emb_dim,
        num_cell_types=NUM_CELL_TYPES if NUM_CELL_TYPES > 0 else None
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    trainer = PerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        device=device,
        cond_label_lookup=cond_label_lookup
    )

    print(f"\nLoading checkpoint from {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    print(f"Model loaded! Trained for {trainer.epoch + 1} epochs.")
    print(f"\nGenerating {args.num_samples_per_pert} samples per perturbation")
    print(f"Sampling parameters: num_steps={args.num_steps}, temperature={args.temperature}")

    model.eval()

    # Enable BF16 for faster inference
    use_amp = True
    amp_dtype = torch.bfloat16
    print(f"Using AMP for inference: {use_amp}, dtype: bfloat16\n")
    
    sampler = PerturbationEulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=args.num_steps,
        temperature=args.temperature,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype
    )

    # Storage for generated cells
    all_generated = []
    all_pert_indices = []
    all_pert_names = []

    with torch.no_grad():
        for pert_name, pert_idx in tqdm(perturbations, desc="Generating"):
            for sample_idx in range(args.num_samples_per_pert):
                x_init = torch.full((1, NUM_GENES),fill_value=graph.mask_index,dtype=torch.long,device=device)

                # Create perturbation label tensor
                pert_label = torch.tensor([pert_idx], dtype=torch.long, device=device)

                # Apply conditional label lookup if available (converts index to precomputed embedding)
                pert_label = trainer._apply_cond_label_lookup(pert_label)

                # Create cell type label tensor if specified
                cell_type_label = None
                if cell_type_idx is not None:
                    cell_type_label = torch.tensor([cell_type_idx], dtype=torch.long, device=device)

                generated = sampler.sample(
                    x_init,
                    pert_labels=pert_label,
                    cell_type_labels=cell_type_label,
                    show_progress=False
                )
                
                all_generated.append(generated.cpu())
                all_pert_indices.append(pert_idx)
                all_pert_names.append(pert_name)

    all_generated = torch.cat(all_generated, dim=0)  # [num_total_samples, num_genes]
    print(f"\nGenerated {len(all_generated)} cells total")

    metadata = {
        "num_cells": len(all_generated),
        "num_perturbations": len(perturbations),
        "num_samples_per_pert": args.num_samples_per_pert,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "perturbations": [p[0] for p in perturbations],
        "checkpoint": str(checkpoint_path),
        "cell_type": args.cell_type,
        "cell_type_idx": cell_type_idx,
    }
    with open(output_dir / "generation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create obs dictionary with cell type information
    obs_dict = {
        'perturbation': all_pert_names,
        'perturbation_idx': all_pert_indices,
        'sample_idx': [i % args.num_samples_per_pert for i in range(len(all_generated))]
    }
    
    # Add cell type to obs if specified
    if args.cell_type:
        obs_dict['cell_type'] = [args.cell_type] * len(all_generated)
        obs_dict['cell_type_idx'] = [cell_type_idx] * len(all_generated)
    
    adata_generated = sc.AnnData(
        X=all_generated.numpy(),
        obs=obs_dict
    )
    adata_generated.write_h5ad(output_dir / "generated_cells.h5ad")

    # Generate summary statistics
    print("\n" + "="*50)
    print("Generation Summary")
    print("="*50)
    
    for pert_name, pert_idx in perturbations:
        mask = np.array(all_pert_indices) == pert_idx
        cells = all_generated[mask]
        
        mean_expr = cells.float().mean().item()
        std_expr = cells.float().std().item()
        sparsity = (cells == 0).float().mean().item()
        
        print(f"{pert_name}:")
        print(f"  Samples: {mask.sum()}")
        print(f"  Mean expression: {mean_expr:.2f}")
        print(f"  Std expression: {std_expr:.2f}")
        print(f"  Sparsity: {sparsity:.2%}")

    # Visualizations
    print("\nGenerating visualizations...")
    
    # 1. Distribution of expression values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(all_generated.flatten().numpy(), bins=50, alpha=0.7)
    axes[0].set_xlabel('Expression Bin')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Generated Expression Values')
    
    
    print_success(output_dir)


if __name__ == "__main__":
    main()