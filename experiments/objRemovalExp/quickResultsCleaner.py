import os
import shutil

removal_path = "/n/fs/obj-cv/experiment_project/savedRemovals/kitchen94"

for scene_id in sorted(os.listdir( removal_path )):
    if scene_id == "logs" or scene_id == "infoFolder":
        continue
    if not os.path.exists( os.path.join( removal_path, scene_id, "Normal_Image.png" ) ):
        print("To be removed: ", scene_id)

confirm = input("Removal stuff?")
if confirm == "y":
    for scene_id in sorted(os.listdir( removal_path )):
        if scene_id == "logs" or scene_id == "infoFolder":
            continue
        if not os.path.exists( os.path.join( removal_path, scene_id, "Normal_Image.png" ) ):
            print("Removing: ", scene_id)
            shutil.rmtree( os.path.join(removal_path, scene_id) )