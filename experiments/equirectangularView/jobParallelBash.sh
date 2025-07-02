# run this from equirectangularView to do everything on DATASET parallelized
# bascially just calls jobParallel.slurm
# 1) CHANGE DATASET name in jobParallel.slurm as well
# 2) make sure equirectangularRender.py uses OPTION 2C
# 3) Optional: Delete old logs folder
# 4) Make sure results (output) folder exists (not sure if it will automatically create)

DATASET="/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94"
# count the immediate subdirectories:
scene_count=$(ls -1d "$DATASET"/*/ 2>/dev/null | wc -l)

# submit an array from 0 to scene_count-1:
sbatch --array=0-$((scene_count-1)) jobParallel.slurm