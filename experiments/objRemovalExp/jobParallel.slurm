#!/bin/bash
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1          # the number of GPUs requested
#SBATCH --mem=24G              # memory
#SBATCH --nice=10000
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH -t 10:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --array=0-99

#SBATCH --mail-user=jw7893@princeton.edu


# --- begin conda setup ---
# replace this path with wherever your conda is installed
CONDA_BASE=/n/fs/obj-cv/miniconda3  
source "$CONDA_BASE/etc/profile.d/conda.sh"
# now you can safely activate
conda activate experiment_project_env
# --- end conda setup ---

# put the command you want to run here

DATASET="/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94"
OUTPUT="/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/results2"

# Change to dataset folder
cd "$DATASET"

# Grab only immediate subfolders, strip trailing slash, and read into array
mapfile -t scenes < <(ls -1d */ 2>/dev/null | sed 's:/$::')

# check how many scenes found
echo "Found ${#scenes[@]} scenes."

# Pick the one match slurm task ID
scene_id="${scenes[$SLURM_ARRAY_TASK_ID]}"

# print
echo "[$SLURM_ARRAY_TASK_ID] to $scene_id"
echo "Blend file: $scene_blend"

# run script
cd /n/fs/obj-cv/experiment_project/experiments/objRemovalExp
python removeObjectsParams.py \
    --dataset "$DATASET" \
    --id "$scene_id" \
    --out "$OUTPUT"