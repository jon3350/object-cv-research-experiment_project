import bpy
import math
from mathutils import Vector, Euler
import os
import sys
import shutil
import time
import datetime
import argparse

def render_and_log(render_file_path):
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still = True)

def apply_scene(blender_scene_path, scene_id, output_folder):
    # open blender scene
    bpy.ops.wm.open_mainfile(filepath = blender_scene_path)
    print(flush=True)

    # configure GPU
    config_GPU_and_render_quality()

    # get active camera
    cam = bpy.context.scene.camera

    # set equirectangular render settings
    print("Old camera type, panorama_type", cam.data.type, cam.data.panorama_type)
    cam.data.type = "PANO"
    cam.data.panorama_type = 'EQUIRECTANGULAR'
    print("New camera type, panorama_type", cam.data.type, cam.data.panorama_type)

    render_and_log(os.path.join(output_folder, scene_id ))

def apply_scene_both_views(blender_scene_path, scene_id, output_folder, original_img):
    # create the new folder that packages both the original and equirectangular image
    bundle_folder_path = os.path.join(output_folder, scene_id)
    # the scene id argument is when the image gets named
    # (blender is able to create the directory and automatically add .png)
    apply_scene(blender_scene_path, "Equirect_Image", bundle_folder_path)
    # now copy the original image
    shutil.copy(original_img, os.path.join(bundle_folder_path, "Normal_Image.png"))


def config_GPU_and_render_quality():
    # ──────────── Make sure GPU is used ─────────────
    # 3.1 Set render engine
    bpy.context.scene.render.engine = 'CYCLES'

    # 3.2 Configure Cycles backend
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"   # or "OPTIX", "ONEAPI", "HIP"
    prefs.get_devices()                  # populate prefs.devices

    # 3.3 Enable all non-CPU devices
    for dev in prefs.devices:
        if dev.type != 'CPU':
            dev.use = True

    # 3.4 Tell the scene to use GPU
    bpy.context.scene.cycles.device = 'GPU'

    # ──────────── 4) Render settings ─────────────
    # 8192 standard but we will use 512 for testing
    bpy.context.scene.cycles.samples            = 8192
    bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.cycles.samples            = 256
    # bpy.context.scene.render.resolution_percentage = 25

    # ──────────── 5) Print scene & device info ─────────────
    print("Scene name:", bpy.context.scene.name)
    print("Engine      :", bpy.context.scene.render.engine)
    print("Backend     :", prefs.compute_device_type)
    print("Scene device:", bpy.context.scene.cycles.device)
    print("Available devices:")
    for d in prefs.devices:
        print(f"  - {d.name:25} type={d.type:4} use={d.use}")
    # ──────────── Rest of code ─────────────


if __name__ == "__main__":
    # path as to where the scenes data sets are
    dataset_path = "/n/fs/obj-cv/experiment_project/experiments/blenderTransformsTester/TrimmedDataset1"
    # or if you just want to do it on one scene id use hardcode_scene_id_path and hardcode_scene_id
    hardcode_scene_id_path = "/n/fs/obj-cv/experiment_project/experiments/blenderTransformsTester/TrimmedDataset1/112d1667/scene.blend"
    hardcode_scene_id = "112d1667"
    # output folder
    output_folder = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView/results"

    # Option 1 - store equirectangularRenders in results. do NOT store original png
    # for scene_id in os.listdir(dataset_path):
    #     print("Scene_id:", scene_id)
    #     scene_path = os.path.join(dataset_path, scene_id, "scene.blend")
    #     print(scene_path, flush=True)
    #     apply_scene(scene_path, scene_id)

    # Option 1b - do a single scene_id instead of a folder dataset
    # print("Scene_id_path:", hardcode_scene_id_path)
    # print("Scene_id:", hardcode_scene_id)
    # apply_scene(hardcode_scene_id_path, hardcode_scene_id)

    # Option 2 - store original png and equirectangularRenders
    # print("Start: ", datetime.datetime.now())
    # dataset_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/livingRoom99"
    # output_folder = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView/results"
    # for scene_id in os.listdir(dataset_path):
    #     # exclude non-directories
    #     if not os.path.isdir( os.path.join(dataset_path, scene_id) ):
    #         print("Not a dir:", scene_id)
    #         continue
    #     # skip folders that have already been created (from previous incomplete runs)
    #     if os.path.isdir( os.path.join(output_folder, scene_id) ):
    #         print("Created Already:", scene_id)
    #         continue
        
    #     print("TimestampBefore: ", datetime.datetime.now())
    #     print("Scene_id:", scene_id)
    #     blender_scene_path = os.path.join(dataset_path, scene_id, "fine", "scene.blend")
    #     original_img = os.path.join(dataset_path, scene_id, "frames", "Image", "camera_0", "Image_0_0_0048_0.png")
    #     apply_scene_both_views(blender_scene_path, scene_id, output_folder, original_img)
    #     print("TimestampAfter: ", datetime.datetime.now())
    # print("End: ", datetime.datetime.now())


    # Option 2b - do on a single image
    # blender_scene_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/diningRoom99/6bde512c/fine/scene.blend"
    # scene_id = "6bde512c"
    # output_folder = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView/results_v1"
    # original_img = "/n/fs/obj-cv/infinigen_project/savedDatasets/dinningRoom99/6bde512c/frames/Image/camera_0/Image_0_0_0048_0.png"
    # apply_scene_both_views(blender_scene_path, scene_id, output_folder, original_img)

    # Option 2c - do a single item with arg parser
    print("Start: ", datetime.datetime.now())
    parser = argparse.ArgumentParser(description = "Equirectangular Renderer")
    parser.add_argument("--blend", required = True, help="blender_scene_path")
    parser.add_argument("--id", required = True, help="scene_id")
    parser.add_argument("--out", required = True, help="output_folder")
    parser.add_argument("--ori", required = True, help = "original_img")
    args = parser.parse_args()

    blender_scene_path = args.blend
    scene_id = args.id
    output_folder = args.out
    original_img = args.ori

    # skip folders that have already been created (from previous incomplete runs)
    if os.path.isdir( os.path.join(output_folder, scene_id) ):
        print("Created Already:", scene_id)
        exit()
    
    apply_scene_both_views(blender_scene_path, scene_id, output_folder, original_img)
    print("End: ", datetime.datetime.now())
