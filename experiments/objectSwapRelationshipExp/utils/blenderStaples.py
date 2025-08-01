# provides functions to use GPU and render
"""
render_and_log( render_file_path ):
Renders the current scene to the destination file path

config_GPU_and_render_quality():
Must call this after a new scene is open to make sure GPU is being used
"""

import bpy
import os

#----------------------------------------------------------------------
# BLENDER STAPLES
def render_and_log(render_file_path):
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still = True)

def config_GPU_and_render_quality(samples = 1024, resolution_percentage = 50):
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
    bpy.context.scene.cycles.samples            = samples
    bpy.context.scene.render.resolution_percentage = resolution_percentage
    # 8192 standard but we will use 512 for testing
    # bpy.context.scene.cycles.samples            = 8192
    # bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.cycles.samples            = 1024
    # bpy.context.scene.render.resolution_percentage = 50
    # bpy.context.scene.cycles.samples            = 256
    # bpy.context.scene.render.resolution_percentage = 25
    # bpy.context.scene.cycles.samples            = 32
    # bpy.context.scene.render.resolution_percentage = 5

    # ──────────── 5) Print scene & device info ─────────────
    print("Scene name:", bpy.context.scene.name)
    print("Engine      :", bpy.context.scene.render.engine)
    print("Backend     :", prefs.compute_device_type)
    print("Scene device:", bpy.context.scene.cycles.device)
    print("Available devices:")
    for d in prefs.devices:
        print(f"  - {d.name:25} type={d.type:4} use={d.use}")
    # ──────────── Rest of code ─────────────
#----------------------------------------------------------------------


# unit testing
if __name__ == "__main__":
    blender_scene_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94/1e122544/fine/scene.blend"
    output_folder = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/utils"
    img_name = "test_image"

    # Open .blend file
    # bpy.ops.wm.open_mainfile(filepath = blender_scene_path )
    bpy.ops.wm.open_mainfile(filepath = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/utils/bedroom1.blend" )

    # configure GPU
    config_GPU_and_render_quality( samples=8192, resolution_percentage=100 )

    # blender automatically adds the .png to img_name
    render_and_log( os.path.join( output_folder, img_name ) )
    