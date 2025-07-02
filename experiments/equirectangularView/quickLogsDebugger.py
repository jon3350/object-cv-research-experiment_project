# Only 98 folders were created by equirectangularRender when 99 were expected
# fine the file that contains an error message

import os

logs_path = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView/logs"

for file in os.listdir(logs_path):
    if os.path.getsize( os.path.join(logs_path, file) ) != 0 and "err" in file:
        print(file)