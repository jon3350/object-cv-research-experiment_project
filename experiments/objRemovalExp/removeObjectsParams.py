import bpy
import transformObjects
import os
import json
import re
import argparse
import shutil
import numpy as np
import sys

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
    bpy.context.scene.cycles.samples            = 8192
    bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.cycles.samples            = 2048
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


if __name__ == "__main__":
    dataset_path = "/n/fs/obj-cv/infinigen_project/Backups/CopyBackups/bedroom10"
    scene_id = "5b180cde"
    output_folder = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/results"

    parser = argparse.ArgumentParser(description = "removeObjectsParams")
    parser.add_argument("--dataset", required = True, help="dataset_path")
    parser.add_argument("--id", required = True, help="scene_id")
    parser.add_argument("--out", required = True, help="output_folder")
    args = parser.parse_args()

    dataset_path = args.dataset
    scene_id = args.id
    output_folder = args.out

    # skips things that have already been done already
    if os.path.isdir( os.path.join(output_folder, f"{scene_id}" ) ):
        print("Output dir with scene id already exists. exiting...")
        print("Output dir with scene id already exists. exiting...", file=sys.stderr)
        exit()

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

    # Conversion_dicts
    # dicts to convert between object-indicies to solve_state_json_names and blender_names
    indexToBlender = {}
    blenderToIndex = {}
    jsonToBlender = {}
    blenderToJson = {}

    # get all object indecies from the objectsegmentation array
    # frames/ObjectSegmentation/camera_0/ObjectSegmentation_0_0_0048_0.npy
    obj_seg_path = os.path.join( dataset_path, scene_id, "frames", "ObjectSegmentation", "camera_0", "ObjectSegmentation_0_0_0048_0.npy")
    obj_seg_arr = np.load(obj_seg_path)
    obj_seg_set = set( obj_seg_arr.flatten() )
    if 0 in obj_seg_set:
        obj_seg_set.remove(0) # remove 0 from set

    # go through objects.json, find all matching objects, and update the conversion_dicts
    obj_json_path = os.path.join( dataset_path, scene_id, "frames", "Objects", "camera_0", "Objects_0_0_0048_0.json")
    with open( obj_json_path, "r") as obj_json_file:
        obj_json_map = json.load(obj_json_file)
    
    for key, meta in obj_json_map.items():
        blender_obj_name = key
        obj_index = meta["object_index"]
        if obj_index in obj_seg_set:
            indexToBlender[obj_index] = blender_obj_name
            blenderToIndex[blender_obj_name] = obj_index
    
    print("PRE-FILTERING: Blender-Index Conversions: ", "Number of enteries: ", len(indexToBlender))
    print("IndexToBlender: ", indexToBlender)
    print("BlenderToIndex: ", blenderToIndex)
    print("\n", flush = True)

    # filter out unwanted objects
    filtered_list = [] # a list of (obj, index) to be removed
    filter_set = {"kitchen_0", "living-room_0", "dining-room_0", "bedroom_0", "bathroom_0"}
    for blender_obj_name, obj_index in blenderToIndex.items():
        for filter_item in filter_set:
            if filter_item in blender_obj_name:
                filtered_list.append( (blender_obj_name, obj_index) )
                break

    for blender_obj_name, obj_index in filtered_list:
        blenderToIndex.pop(blender_obj_name)
        indexToBlender.pop(obj_index)

    # populate dicts for blender_obj_names to and from solve_state_json_obj_names
    solve_state_json_path = os.path.join( dataset_path, scene_id, "coarse", "solve_state.json" )
    with open(solve_state_json_path, "r") as solve_state_json_file:
        solve_state_json_map = json.load(solve_state_json_file)
        solve_state_json_map = solve_state_json_map["objs"] # unpack objs
    for key, meta in solve_state_json_map.items():
        if meta['obj'] in blenderToIndex:
            json_obj_name = key
            blender_obj_name = meta['obj']
            blenderToJson[blender_obj_name] = json_obj_name
            jsonToBlender[json_obj_name] = blender_obj_name
    # json has less objects than index and blender
    for obj_blender_name in blenderToIndex.keys():
        if obj_blender_name not in blenderToJson:
            blenderToJson[obj_blender_name] = None

    print("Filter: ", "removedCount:", len(filtered_list), "remainingCount", len(blenderToIndex))
    print("removed: ", filtered_list)
    print("remaining: ", blenderToIndex.keys())
    print("POST-FILTER: (index: blender : json)")
    for obj_index, blender_obj_name in indexToBlender.items():
        print(f"{obj_index}: {blender_obj_name} : {blenderToJson[blender_obj_name]}")
    print("\n", flush = True)


    # for each object (blender_name), map it to a list of 'dependents' that depend on it
    # i.e. table should map to all the plates on the table
    # use blender names for objects
    dependents_map = {}

    for root in blenderToIndex.keys():
        # run bfs
        queue = [ root ]
        dependents = set() # fill it with all objects connected to parent
        # TICKET CHANGED THIS FROM set(root) NEVER RAN TO TEST
        visited = { root } # technically not needed since you can only depend on one thing and there are no cycles
        while queue:
            # process current object by adding it to dependents set
            curr_obj_blender = queue.pop(0)
            if curr_obj_blender != root: # don't include the root as a dependent
                dependents.add( curr_obj_blender )

            # put all eligible children in the queue

            # Objects.json condition...
            # accept children of curr_obj_blender
            for child_index in obj_json_map[curr_obj_blender]["children"]:
                if child_index in indexToBlender and indexToBlender[child_index] not in visited:
                    visited.add( indexToBlender[child_index] )
                    queue.append( indexToBlender[child_index] )
            
            # Solve_state.json condition...
            if not blenderToJson[curr_obj_blender]: # some blender objs are not listed in obj
                continue
            for key in jsonToBlender.keys():
                if jsonToBlender[key] in visited:
                    continue
                for rel in solve_state_json_map[key]["relations"]:
                    target_name = rel["target_name"]
                    relation_type = rel["relation"]["relation_type"]
                    child_tags = rel["relation"]["child_tags"]
                    if (target_name == blenderToJson[curr_obj_blender]
                            and relation_type == "StableAgainst"
                            and "Subpart(bottom)" in child_tags
                    ):
                        visited.add( jsonToBlender[key] )
                        queue.append( jsonToBlender[key] )
        dependents_map[ root ] = dependents

    # print out pretty formatted dependents_map
    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    pretty_print = json.dumps( dependents_map, indent=2, default=set_default )
    print( "Dependents_map: ", pretty_print )

    # categorize objects by groups
    # create maps from groups to objects and vice versa
    groupToObjects = {}
    objectToGroup = {}
    for blender_obj_name in blenderToIndex.keys():
        index = blender_obj_name.find("Factory")
        if index != -1:
            group_name = blender_obj_name[:index]
        else:
            group_name = blender_obj_name
        objectToGroup[blender_obj_name] = group_name
        if group_name in groupToObjects:
            groupToObjects[group_name].append(blender_obj_name)
        else:
            groupToObjects[group_name] = [ blender_obj_name ]

    # a dictionary to save important info that will be written to an output file later
    file_dict = {}
    

    # remove groups of objects (ie all chairs instead of just one chair)
    for group in groupToObjects.keys():
        # don't do groups of 1 since we will do that below for single removals
        if len(groupToObjects[group]) == 1:
            continue
        old_flags = transformObjects.remove_object_blender_name(groupToObjects[group][0])
        for target_obj in groupToObjects[group]:
            transformObjects.remove_object_blender_name(target_obj)
            for dependent_obj in dependents_map[target_obj]:
                transformObjects.remove_object_blender_name(dependent_obj)
        image_name = f"{group}"
        render_and_log(os.path.join(output_folder, f"{scene_id}", image_name ))
        transformObjects.restore_object(old_flags)
        # record details in file_dict
        file_dict[image_name] = {
                "scene_id": scene_id,
                "isNormalImg": False,
                "removalGroup": group,
                "removalType": "group",
                "removalTargets": [ {"parent": blender_obj_name,
                                    "dependents": list(dependents_map[blender_obj_name])
                                    } for blender_obj_name in groupToObjects[group]
                                    ]
        }

    # remove each object one by one and generate an according image
    for blender_obj_name in blenderToIndex.keys():
        old_flags = transformObjects.remove_object_blender_name(blender_obj_name)
        # remove all 'dependents' as well
        for dependent_obj in dependents_map[blender_obj_name]:
            transformObjects.remove_object_blender_name(dependent_obj)
        image_name = f"{blender_obj_name}"
        render_and_log(os.path.join(output_folder, f"{scene_id}", image_name ))
        transformObjects.restore_object(old_flags)
        # record details in file_dict
        file_dict[image_name] = {
            "scene_id": scene_id,
            "isNormalImg": False,
            "removalGroup": objectToGroup[blender_obj_name],
            "removalType": "single",
            "removalTargets": [ {"parent": blender_obj_name,
                                 "dependents": list(dependents_map[blender_obj_name])
                                 }
                                ]
        }


    # add the original image
    original_img = os.path.join( dataset_path, scene_id, "frames", "Image", "camera_0", "Image_0_0_0048_0.png" )
    shutil.copy(original_img, os.path.join(output_folder, f"{scene_id}", "Normal_Image.png"))
    # record details in file_dict
    file_dict["Normal_Image.png"] = {
            "scene_id": scene_id,
            "isNormalImg": True,
            "removalGroup": "None",
            "removalType": "None",
            "removalTargets": [ ]
    }



    # write file_dict to file
    if not os.path.isdir( os.path.join(output_folder, "infoFolder") ):
        os.makedirs( os.path.join(output_folder, "infoFolder"), exist_ok=True )
    with open( os.path.join( output_folder, "infoFolder", f"Info{scene_id}.json"), "w" ) as info_file:
        json.dump( file_dict, info_file, indent=2, default=set_default )