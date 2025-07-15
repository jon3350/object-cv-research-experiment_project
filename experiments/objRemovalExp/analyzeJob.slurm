#!/bin/bash
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0          # the number of GPUs requested
#SBATCH --mem=24G              # memory
#SBATCH --nice=10000
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH -t 1:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --array=0-4

#SBATCH --mail-user=jw7893@princeton.edu


# --- begin conda setup ---
# replace this path with wherever your conda is installed
CONDA_BASE=/n/fs/obj-cv/miniconda3  
source "$CONDA_BASE/etc/profile.d/conda.sh"
# now you can safely activate
conda activate experiment_project_env
# --- end conda setup ---

# put the command you want to run here
declare -a removal_array=(
    [0]=/n/fs/obj-cv/experiment_project/savedRemovals/bathroom97
    [1]=/n/fs/obj-cv/experiment_project/savedRemovals/bedroom96
    [2]=/n/fs/obj-cv/experiment_project/savedRemovals/diningroom99
    [3]=/n/fs/obj-cv/experiment_project/savedRemovals/kitchen94
    [4]=/n/fs/obj-cv/experiment_project/savedRemovals/livingroom99
)
declare -a dataset_array=(
    [0]=/n/fs/obj-cv/infinigen_project/savedDatasets/bathroom97
    [1]=/n/fs/obj-cv/infinigen_project/savedDatasets/bedroom96
    [2]=/n/fs/obj-cv/infinigen_project/savedDatasets/diningRoom99
    [3]=/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94
    [4]=/n/fs/obj-cv/infinigen_project/savedDatasets/livingRoom99
)
declare -a write_array=(
    [0]=/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/bathroom97Data
    [1]=/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/bedroom96Data
    [2]=/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/diningroom97Data
    [3]=/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/kitchen94Data
    [4]=/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/livingroom97Data
) 
declare -a ground_id=(
    [0]=45
    [1]=52
    [2]=121
    [3]=203
    [4]=215
)

# Pick the one match slurm task ID
output_folder="${removal_array[$SLURM_ARRAY_TASK_ID]}"
dataset_path="${dataset_array[$SLURM_ARRAY_TASK_ID]}"
data_write_folder="${write_array[$SLURM_ARRAY_TASK_ID]}"
ground_id="${ground_id[$SLURM_ARRAY_TASK_ID]}"

# print
echo "[$SLURM_ARRAY_TASK_ID]"
echo "${output_folder}"
echo "${dataset_path}"
echo "${data_write_folder}"
echo "${ground_id}"


# run script
cd /n/fs/obj-cv/experiment_project/experiments/objRemovalExp
python analyzeData.py \
    --output_folder "${output_folder}" \
    --dataset_path "${dataset_path}" \
    --data_write_folder "${data_write_folder}" \
    --ground_id "${ground_id}"