#!/bin/bash
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4          # the number of GPUs requested
#SBATCH --mem=24G              # memory
#SBATCH --output=results/logs/slurm17to50.out
#SBATCH -t 5:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email

#SBATCH --mail-user=jw7893@princeton.edu


# --- begin conda setup ---
# replace this path with wherever your conda is installed
CONDA_BASE=/n/fs/obj-cv/miniconda3  
source "$CONDA_BASE/etc/profile.d/conda.sh"
# now you can safely activate
conda activate experiment_proj_clone
# --- end conda setup ---

# put the command you want to run here
cd /n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp
python swapObjectsMiniSolver.py