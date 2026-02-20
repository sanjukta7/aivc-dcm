#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --gpus-per-node=
#SBATCH --mem-per-gpu=
#SBATCH --time=
#SBATCH --qos=
#SBATCH --output=slurm_out/
#SBATCH --mail-type=END,FAIL

# SLURM batch script for training SEDD Perturbation-seq model using YAML config
#
# Usage:
#   sbatch bash/train_pseq.sbatch --config configs/perturbseq_small.yaml --train_data_path datasets/1M/1m_train_processed.h5ad --cond_labels_pt_path datasets/1M/cytokines.pt
#
# Or with environment variables:
#   CONFIG=configs/perturbseq_small.yaml TRAIN_DATA_PATH=datasets/1M/1m_train_processed.h5ad COND_LABELS_PT_PATH=datasets/1M/cytokines.pt sbatch bash/train_pseq.sbatch

# Change to project directory
cd || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Load CUDA module
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

# Activate virtual environment (adjust path as needed)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Memory optimization for large sequences
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check CUDA availability
echo "CUDA check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "Could not check CUDA"

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Default parameters (can be overridden by environment variables)
CONFIG="${CONFIG:-configs/perturbseq_small.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
COND_LABELS_PT_PATH="${COND_LABELS_PT_PATH:-}"

# Print training configuration
echo "Training Configuration:"
echo "Config file: $CONFIG"
if [ -n "$TRAIN_DATA_PATH" ]; then
    echo "Data path: $TRAIN_DATA_PATH"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    echo "Cond labels .pt: $COND_LABELS_PT_PATH"
fi
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Build the command
CMD="python scripts/train_perturbseq.py --config $CONFIG"

# Add data path if provided
if [ -n "$TRAIN_DATA_PATH" ]; then
    CMD="$CMD --train_data_path $TRAIN_DATA_PATH"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    CMD="$CMD --cond_labels_pt_path $COND_LABELS_PT_PATH"
fi

# Add any additional command line arguments passed to sbatch
CMD="$CMD $@"

# Run training with srun
echo "Running command: $CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="