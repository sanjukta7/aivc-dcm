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
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import SEDDTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import SEDDTrainer
from sedd.data import train_val_split
from sedd.sampling import EulerSampler

# Import TOML library
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(config_path="config.toml"):
    """Load configuration from TOML file."""
    config_file = Path(__file__).parent.parent / config_path
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_file}, using defaults")
        return {}

    with open(config_file, "rb") as f:
        return tomllib.load(f)


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
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    raise ValueError(f"No checkpoints found in {experiment_dir}")


def parse_args():
    # Load config file
    config = load_config()
    data_config = config.get("data", {})

    parser = argparse.ArgumentParser(description="Generate new cells with trained SEDD model")

    # Experiment/checkpoint arguments
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

    # Data arguments (for comparison with real cells)
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_config.get("test_data", None) or data_config.get("train_data", None),
        help="Path to h5ad file containing real data for comparison (defaults to config.toml)"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=data_config.get("val_fraction", 0.1),
        help="Validation fraction (used if test_data not separate)"
    )
    parser.add_argument(
        "--use_train_split",
        action="store_true",
        help="Use training split instead of validation split for comparison"
    )

    # Generation arguments
    parser.add_argument(
        "--num_generate",
        type=int,
        default=100,
        help="Number of cells to generate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of sampling steps for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )

    # Visualization arguments
    parser.add_argument(
        "--num_cells_visualize",
        type=int,
        default=3,
        help="Number of generated cells to visualize"
    )
    parser.add_argument(
        "--num_real_visualize",
        type=int,
        default=3,
        help="Number of real cells to visualize for comparison"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


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
    output_dir = experiment_dir / "generation_results"
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

    # Load data for comparison
    if not args.data_path:
        raise ValueError("No data_path provided. Set it in config.toml or pass --data_path")

    print(f"\nLoading data from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Convert to tensor
    expression = adata.X
    dataset = torch.tensor(expression).long()

    # Calculate vocab size
    NUM_BINS = int(dataset.max().item())
    NUM_GENES = dataset.shape[1]
    VOCAB_SIZE = NUM_BINS + 1

    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Split into train/val
    train_dataset, test_dataset = train_val_split(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    # Choose which split to use for comparison
    real_dataset = train_dataset if args.use_train_split else test_dataset
    split_name = "train" if args.use_train_split else "val"
    print(f"Using {split_name} split for comparison: {len(real_dataset)} cells")

    # Get some real cells for comparison
    real_cells = real_dataset[:min(len(real_dataset), args.num_real_visualize)]


    model = SEDDTransformerSmall(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        hidden_dim=train_config.get("hidden_dim", 128),
        num_layers=train_config.get("num_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        dropout=train_config.get("dropout", 0.1),
        max_seq_len=NUM_GENES
    ).to(device)

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    # Create trainer and load checkpoint
    trainer = SEDDTrainer(
        model=model,
        graph=graph,
        noise=noise,
        device=device
    )

    print(f"\nLoading checkpoint from {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    print(f"Model loaded! Trained for {trainer.epoch + 1} epochs.")

    # Create sampler
    print(f"\nGenerating {args.num_generate} cells with {args.num_steps} steps, "
          f"temperature={args.temperature}")

    # Enable BF16 for faster inference
    use_amp = True
    amp_dtype = torch.bfloat16
    print(f"Using AMP for inference: {use_amp}, dtype: bfloat16\n")
    
    sampler = EulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=args.num_steps,
        device=device,
        temperature=args.temperature,
        use_amp=use_amp,
        amp_dtype=amp_dtype
    )

    # Generate cells from all-masked starting point
    x_init = graph.sample_limiting((args.num_generate, NUM_GENES), device)
    print(f"Starting from all-masked state (mask token: {x_init[0, 0].item()})")

    generated = sampler.sample(x_init, show_progress=True)
    print(f"Generated {generated.shape[0]} cells")

    # Save generated cells
    print(f"\nSaving generated cells to {output_dir}")
    torch.save(generated.cpu(), output_dir / "generated_cells.pt")
    np.save(output_dir / "generated_cells.npy", generated.cpu().numpy())

    # Calculate and compare statistics
    print("\n" + "="*50)
    print("Generation Statistics")
    print("="*50)

    real_batch = real_dataset[:min(len(real_dataset), args.num_generate)].to(device)

    real_mean = real_batch.float().mean(dim=0).cpu().numpy()
    gen_mean = generated.float().mean(dim=0).cpu().numpy()

    real_std = real_batch.float().std(dim=0).cpu().numpy()
    gen_std = generated.float().std(dim=0).cpu().numpy()

    # Mean expression correlation
    mean_corr = np.corrcoef(real_mean, gen_mean)[0, 1]
    print(f"Mean expression correlation: {mean_corr:.4f}")

    # Std correlation
    std_corr = np.corrcoef(real_std, gen_std)[0, 1]
    print(f"Std expression correlation: {std_corr:.4f}")

    # Overall statistics
    print(f"\nReal cells - Mean: {real_batch.float().mean():.2f}, Std: {real_batch.float().std():.2f}")
    print(f"Generated cells - Mean: {generated.float().mean():.2f}, Std: {generated.float().std():.2f}")

    # Save metrics
    metrics = {
        "num_generated": args.num_generate,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "mean_correlation": float(mean_corr),
        "std_correlation": float(std_corr),
        "real_mean": float(real_batch.float().mean()),
        "real_std": float(real_batch.float().std()),
        "generated_mean": float(generated.float().mean()),
        "generated_std": float(generated.float().std()),
        "checkpoint": str(checkpoint_path)
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    (output_dir / "mean_corr.txt").write_text(f"{mean_corr:.6f}\n")
    (output_dir / "std_corr.txt").write_text(f"{std_corr:.6f}\n")

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Compare real vs generated cells (bar plots)
    n_real = min(args.num_real_visualize, real_cells.size(0))
    n_gen = min(args.num_cells_visualize, generated.size(0))
    n_cols = max(n_real, n_gen)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)

    # Plot real cells
    for i in range(n_real):
        axes[0, i].bar(range(NUM_GENES), real_cells[i].cpu().numpy(), alpha=0.7, width=1.0)
        axes[0, i].set_xlabel('Gene')
        axes[0, i].set_ylabel('Bin')
        axes[0, i].set_title(f'Real Cell {i+1}')

    # Fill empty slots if needed
    for i in range(n_real, n_cols):
        axes[0, i].axis('off')

    # Plot generated cells
    for i in range(n_gen):
        axes[1, i].bar(range(NUM_GENES), generated[i].cpu().numpy(), alpha=0.7, width=1.0)
        axes[1, i].set_xlabel('Gene')
        axes[1, i].set_ylabel('Bin')
        axes[1, i].set_title(f'Generated Cell {i+1}')

    # Fill empty slots if needed
    for i in range(n_gen, n_cols):
        axes[1, i].axis('off')

    plt.tight_layout()
    fig.savefig(output_dir / "real_vs_generated.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Compare statistics (scatter plots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(real_mean, gen_mean, alpha=0.5)
    axes[0].plot([0, max(real_mean.max(), gen_mean.max())],
                 [0, max(real_mean.max(), gen_mean.max())], 'r--')
    axes[0].set_xlabel('Real Mean Expression')
    axes[0].set_ylabel('Generated Mean Expression')
    axes[0].set_title(f'Mean Expression per Gene (r={mean_corr:.3f})')

    axes[1].scatter(real_std, gen_std, alpha=0.5)
    axes[1].plot([0, max(real_std.max(), gen_std.max())],
                 [0, max(real_std.max(), gen_std.max())], 'r--')
    axes[1].set_xlabel('Real Std Expression')
    axes[1].set_ylabel('Generated Std Expression')
    axes[1].set_title(f'Expression Variance per Gene (r={std_corr:.3f})')

    plt.tight_layout()
    fig.savefig(output_dir / "expression_stats.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall value distribution
    axes[0, 0].hist(real_batch.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 0].hist(generated.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[0, 0].set_xlabel('Expression Bin')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall Expression Distribution')
    axes[0, 0].legend()

    # Mean per cell distribution
    axes[0, 1].hist(real_batch.float().mean(dim=1).cpu().numpy(), bins=30, alpha=0.5, label='Real', density=True)
    axes[0, 1].hist(generated.float().mean(dim=1).cpu().numpy(), bins=30, alpha=0.5, label='Generated', density=True)
    axes[0, 1].set_xlabel('Mean Expression per Cell')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Cell-wise Mean Distribution')
    axes[0, 1].legend()

    # Sparsity (fraction of zeros)
    real_sparsity = (real_batch == 0).float().mean(dim=1).cpu().numpy()
    gen_sparsity = (generated == 0).float().mean(dim=1).cpu().numpy()
    axes[1, 0].hist(real_sparsity, bins=30, alpha=0.5, label='Real', density=True)
    axes[1, 0].hist(gen_sparsity, bins=30, alpha=0.5, label='Generated', density=True)
    axes[1, 0].set_xlabel('Fraction of Zeros')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Sparsity Distribution')
    axes[1, 0].legend()

    # Non-zero mean
    real_nonzero = real_batch.float()
    real_nonzero = real_nonzero[real_nonzero > 0]
    gen_nonzero = generated.float()
    gen_nonzero = gen_nonzero[gen_nonzero > 0]

    axes[1, 1].hist(real_nonzero.cpu().numpy(), bins=50, alpha=0.5, label='Real', density=True)
    axes[1, 1].hist(gen_nonzero.cpu().numpy(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[1, 1].set_xlabel('Expression Bin')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Non-zero Expression Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "distribution_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print(f"Generation complete! Results saved to {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()