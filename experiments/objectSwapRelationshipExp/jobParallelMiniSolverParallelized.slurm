#!/bin/bash
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4          # the number of GPUs requested
#SBATCH --mem=24G              # memory
#SBATCH --output=results/logs/slurm-%A_%a.out
#SBATCH --error=results/logs/slurm-%A_%a.err
#SBATCH -t 00:45:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --array=0-223%9   # MAKE SURE TO MATCH THE INDEX

#SBATCH --mail-user=jw7893@princeton.edu


# --- begin conda setup ---
# replace this path with wherever your conda is installed
CONDA_BASE=/n/fs/obj-cv/miniconda3  
source "$CONDA_BASE/etc/profile.d/conda.sh"
# now you can safely activate
conda activate experiment_proj_clone
# --- end conda setup ---

# put the command you want to run here

DATASET="/n/fs/obj-cv/infinigen_project/savedDatasets/livingRoom99"
OUTPUT="/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/results"
INDEX="$SLURM_ARRAY_TASK_ID"


# print
echo "[$SLURM_ARRAY_TASK_ID]"

# run script
cd /n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp
python swapObjectsMiniSolverParallelized.py \
    --output_folder "$OUTPUT" \
    --dataset_path "$DATASET" \
    --index "$INDEX"