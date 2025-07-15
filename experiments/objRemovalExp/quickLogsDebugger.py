# 2 folders are missing
# find the file that contains an error message

import os

logs_path = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/results/logs"
job_id = "2226150"

# for file in os.listdir(logs_path):
#     if os.path.getsize( os.path.join(logs_path, file) ) != 0 and "err" in file:
#         print(file)

for file in os.listdir(logs_path):
    if "err" not in file or job_id not in file:
        continue
    with open( os.path.join(logs_path, file), "r") as f:
        contents = f.read()
        if (os.path.getsize( os.path.join(logs_path, file) ) != 0 
            and "Output dir with scene id already exists. exiting..." not in contents):
            print(file)