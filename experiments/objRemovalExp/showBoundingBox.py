import bpy
import transformObjects
import os
import json
from mathutils import Vector
import re


def render_and_log(render_file_path):
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still = True)

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
    # bpy.context.scene.cycles.samples            = 8192
    # bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples            = 256
    bpy.context.scene.render.resolution_percentage = 25

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
    dataset_path = "/n/fs/obj-cv/infinigen_project/Backups/CopyBackups/bedroom10"
    scene_id = "5b180cde"
    output_folder = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/results"

    blender_scene_path = os.path.join( dataset_path, scene_id, "fine", "scene.blend" )
    with open( os.path.join( dataset_path, scene_id, "coarse", "solve_state.json" ), "r" ) as f:
        solve_state = json.load(f)

    # open blender scene
    bpy.ops.wm.open_mainfile(filepath = blender_scene_path)
    print(flush=True)

    # configure GPU
    config_GPU_and_render_quality()

    # get active camera
    cam = bpy.context.scene.camera

    # for key in obj_list:
    #     blender_obj = bpy.data.objects[ solve_state["objs"][key]["obj"] ]
    #     bbox = transformObjects.create_bounding_box_cube( blender_obj )
    #     render_and_log(os.path.join(output_folder, f"{scene_id}_{key}" ))
    #     bpy.data.objects.remove(bbox, do_unlink=True)

    # key = "window.004"
    # blender_obj = bpy.data.objects[ solve_state["objs"][key]["obj"] ]
    # bbox = transformObjects.create_bounding_box_cube( blender_obj )
    # for vert in bbox.bound_box:
    #     print(Vector(vert))
    # print(transformObjects.count_vertices(bbox))
    # bbox.scale = (200.0, 200.0, 200.0)
    # bpy.context.view_layer.update()
    # render_and_log(os.path.join(output_folder, f"{scene_id}_{key}" ))
    # bpy.data.objects.remove(bbox, do_unlink=True)

    obj_list = transformObjects.get_objects_in_camera_frame(solve_state)
    cleaned_keys = [ re.sub(r"[^A-Za-z]", "", k) for k in obj_list if "Factory" in k]
    obj_key_set = set(cleaned_keys)
    print(obj_key_set)

    # for obj_group in obj_key_set:

    #     bound_box_list = []

    #     # create the bounding boxes
    #     for json_obj in solve_state["objs"]:
    #         if not obj_group in json_obj:
    #             continue
            
    #         blender_obj = bpy.data.objects[ solve_state["objs"][json_obj]["obj"] ]
    #         bbox = transformObjects.create_bounding_box_cube( blender_obj )
    #         bound_box_list.append(bbox)
    
    #     # render the images
    #     render_and_log(os.path.join(output_folder, f"{obj_group}" ))

    #     # undo stuff
    #     for bbox in bound_box_list:
    #         bpy.data.objects.remove(bbox, do_unlink=True)


    for obj_group in obj_key_set:

        first_old_flags = None

        # delete all relevant objects
        for json_obj in solve_state["objs"]:
            if not obj_group in json_obj:
                continue
            
            blender_obj = bpy.data.objects[ solve_state["objs"][json_obj]["obj"] ]
            old_flags = transformObjects.remove_object(json_obj, True, solve_state)
            if not first_old_flags:
                first_old_flags = old_flags
    
        # render the images
        render_and_log(os.path.join(output_folder, f"a{obj_group}" ))

        # undo stuff
        transformObjects.restore_object(first_old_flags)