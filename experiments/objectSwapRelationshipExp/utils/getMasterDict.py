# imports for master dict
import os
import json
import numpy as np


#----------------------------------------------------------------------
# MASTER_DICT
# get all the important information you need
# that does not require opening the blender file
def getMasterDict(dataset_path, scene_id):
    """
    Given a dataset_path and a scene_id returns a dict containing useful information
    
    Conversion dictionaries:
    'indexToBlender', 'blenderToIndex', 'jsonToBlender', 'blenderToJson'
    index = number used to identify objects in objects.json
    blender = object named used by blender used in objects.json and target of solve_state["objs"][json_name]["obj"]
    json = object named used in solve_state.json
    Not all objects are in solve_state.json so jsonToBlender can have less keys and blenderToJson can map keys to None
    
    Relational Dependencies:
    'dependents_map', 'groupToObjects', 'objectToGroup'
    dependents_maps[table_blender_name] = { pan_on_table_blender_name, ...  }
    groupToObjects['chair'] = ['chairFactory.spawn_asset(1)', 'chairFactory.spawn_asset(2)', ...]
    objectToGroup['chairFactory.spawn_asset(1)'] = 'chair'

    From objectSegmentation.npy:
    'obj_seg_arr'
    obj_seg_arr[x][y] = index of object that occupies pixel (x,y)
    """

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
    
    # print("PRE-FILTERING: Blender-Index Conversions: ", "Number of enteries: ", len(indexToBlender))
    # print("IndexToBlender: ", indexToBlender)
    # print("BlenderToIndex: ", blenderToIndex)
    # print("\n", flush = True)

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

    # print("Filter: ", "removedCount:", len(filtered_list), "remainingCount", len(blenderToIndex))
    # print("removed: ", filtered_list)
    # print("remaining: ", blenderToIndex.keys())
    # print("POST-FILTER: (index: blender : json)")
    # for obj_index, blender_obj_name in indexToBlender.items():
    #     print(f"{obj_index}: {blender_obj_name} : {blenderToJson[blender_obj_name]}")
    # print("\n", flush = True)


    # for each object (blender_name), map it to a list of 'dependents' that depend on it
    # i.e. table should map to all the plates on the table
    # use blender names for objects
    dependents_map = {}

    for root in blenderToIndex.keys():
        # run bfs
        queue = [ root ]
        dependents = set() # fill it with all objects connected to parent
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
    # def set_default(obj):
    #     if isinstance(obj, set):
    #         return list(obj)
    #     raise TypeError
    # pretty_print = json.dumps( dependents_map, indent=2, default=set_default )
    # print( "Dependents_map: ", pretty_print )

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
    
    masterDict = {
        # our 4 name conversion dictionaries
        "indexToBlender": indexToBlender,
        "blenderToIndex": blenderToIndex,
        "jsonToBlender": jsonToBlender,
        "blenderToJson": blenderToJson,

        # relational dependencies
        "dependents_map": dependents_map,
        "groupToObjects": groupToObjects,
        "objectToGroup": objectToGroup,

        "obj_seg_arr": obj_seg_arr, #from objectSegmentation.npy
    }
    return masterDict
#----------------------------------------------------------------------


# unit testing
if __name__=="__main__":
    dataset_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94"
    scene_id = "1e122544"

    mDct = getMasterDict( dataset_path, scene_id )
    

    # helper functions to sterilize sets when json dumps
    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    

    # print the conversion dicts
    print( '\nmDct["indexToBlender"]:' )
    print( json.dumps( mDct["indexToBlender"] ) )

    print( '\nmDct["blenderToIndex"]:' )
    print( json.dumps( mDct["blenderToIndex"] ) )

    print( '\nmDct["jsonToBlender"]:' )
    print( json.dumps( mDct["jsonToBlender"] ) )

    print( '\nmDct["blenderToJson"]:' )
    print( json.dumps( mDct["blenderToJson"] ) )

    # print the relational dicts
    print( '\nmDct["dependents_map"]:' )
    print( json.dumps( mDct["dependents_map"], default=set_default ) )

    print( '\nmDct["groupToObjects"]:' )
    print( json.dumps( mDct["groupToObjects"] ) )

    print( '\nmDct["objectToGroup"]:' )
    print( json.dumps( mDct["objectToGroup"] ) )


    # for object seg array, identify all the unique objects and aggregate the number of pixels
    # create a map from objectIndex to pixel number
    unique_vals, counts = np.unique( mDct["obj_seg_arr"], return_counts=True )
    indexToPixel = dict( zip( unique_vals.tolist(), counts.tolist() ) ) # might contain 0 which indexToBlender doesn't have
    totalPixels = sum(indexToPixel.values())

    print( '\nmDct["obj_seg_arr"].shape():' )
    print( mDct["obj_seg_arr"].shape )

    print( '\nindexToPixel:' )
    print( json.dumps(indexToPixel) )

    print( '\ntotalPixels:' )
    print( json.dumps(totalPixels) )