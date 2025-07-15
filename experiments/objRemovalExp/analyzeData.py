# general imports
import argparse
import sys
import pandas as pd

# imports for master dict
import os
import json
import numpy as np

# imports for placesCNN()
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

#----------------------------------------------------------------------
# MASTER_DICT
# get all the important information you need
# that does not require opening the blender file
def getMasterDict(dataset_path, scene_id):
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


#----------------------------------------------------------------------
# PLACES365_RESNET
class placesCNN():
    def __init__(self):
        """
        saves model, centre_crop, and classes as instance variables
        model and centre_crop are used in predict()
        classes[i] is the class name of the ith class
        """
        # th architecture to use
        arch = 'resnet18'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()


        # load the image transformer
        centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # variables we care about
        self.model= model
        self.centre_crop = centre_crop
        self.classes = classes
    
    def predict(self, img_path):
        """
        Returns h_x, probs, idx
        h_x[i] is model's prediction for class i
        probs[i] is the model's prediction for the ith highest class
        idx[i] is the class index corresponding to the ith highest class
        """
        img = Image.open(img_path).convert("RGB")
        input_img = V(self.centre_crop(img).unsqueeze(0)) # preprocess & batchify

        # forward pass
        logit = self.model(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        return h_x, probs, idx
#----------------------------------------------------------------------



#----------------------------------------------------------------------
def analyze_scene(output_folder, dataset_path, scene_id, CNN, data_table):
    """
    Given the output_folder, dataset_path, scene_id, CNN, and data_table

    Adds rows to the data_table for every image in the scene. ie Mutates data_table

    Each row corresponds to the CNN models prediction on each image in the scene
    """
    # master dict
    masterDict = getMasterDict( dataset_path, scene_id )
    indexToBlender = masterDict["indexToBlender"]
    blenderToIndex = masterDict["blenderToIndex"]
    jsonToBlender = masterDict["jsonToBlender"]
    blenderToJson = masterDict["blenderToJson"]
    dependents_map = masterDict["dependents_map"]
    groupToObjects = masterDict["groupToObjects"]
    objectToGroup = masterDict["objectToGroup"]
    obj_seg_arr = masterDict["obj_seg_arr"]

    # create a map from objectIndex to pixel number
    unique_vals, counts = np.unique( obj_seg_arr, return_counts=True )
    indexToPixel = dict( zip( unique_vals.tolist(), counts.tolist() ) ) # might contain 0 which indexToBlender doesn't have
    totalPixels = sum(indexToPixel.values())

    # get info about the normal image
    norm_img_path = os.path.join( output_folder, scene_id, "Normal_Image.png" )
    h_x, probs, idx = CNN.predict( norm_img_path )
    norm_ground_prob = h_x[ground_id]
    norm_orig_prob = probs[0]
    norm_orig_idx = idx[0]
    norm_orig_name = CNN.classes[norm_orig_idx]

    # go through every image (except the normal image)
    for img in sorted(os.listdir( os.path.join( output_folder, scene_id ) )):
        if img == "Normal_Image.png":
            continue

        row_dict = {
            "imgName": img,
            "sceneId": scene_id,
            "modelGroundProb": None,
            "modelOrigProb": None,
            "normGroundProb": norm_ground_prob,
            "normOrigProb": norm_orig_prob,
            "normOrigIdx": norm_orig_idx,
            "normOrigName": norm_orig_name,
            "deltaGroundProb": None, # model - norm; ie negative means important
            "deltaOrigProb": None,
            "groupName": None,
            "isSingle": None, # single and all can be true at the same time
            "isAll": None,
            "targetPixels": None,
            "cumPixels": None,
            "cumPixelsPercent": None
        }

        h_x, probs, idx = CNN.predict( os.path.join( output_folder, scene_id, img) )
        row_dict["modelGroundProb"] = h_x[ground_id]
        row_dict["modelOrigProb"] = h_x[norm_orig_idx]

        row_dict["deltaGroundProb"] = h_x[ground_id] - norm_ground_prob
        row_dict["deltaOrigProb"] = h_x[norm_orig_idx] - norm_orig_prob

        # group name, single, all
        # if the image name is the group name
        img_no_png = img[: len(img) - 4 ] # slice out the .png in the img name
        if img_no_png in groupToObjects:
            row_dict["groupName"] = img_no_png
            # rare cases where there is only a single object but it has the group name ie "skirtingboard_support"
            if len( groupToObjects[ img_no_png ] ) == 1:
                row_dict["isSingle"] = True
                row_dict["isAll"] = True
            else:
                row_dict["isSingle"] = False
                row_dict["isAll"] = True
        # if the image name is an object name
        elif img_no_png in objectToGroup:
            row_dict["groupName"] = objectToGroup[img_no_png]
            # if the group contains exactly one object then we have removed a single item
            # and all of them at the same time
            if len( groupToObjects[ objectToGroup[img_no_png] ] ) == 1:
                row_dict["isSingle"] = True
                row_dict["isAll"] = True
            else:
                row_dict["isSingle"] = True
                row_dict["isAll"] = False
        else:
            print("Error: image name is neither an object name or a group name", file=sys.stderr)
            exit()
        
        # targetpixels, cumulativePixels, cumlativePixelsPercent
        row_dict["targetPixels"] = 0
        row_dict["cumPixels"] = 0
        for obj in groupToObjects[ row_dict["groupName"] ]:
            row_dict["targetPixels"] += indexToPixel[ blenderToIndex[ obj ] ]
            row_dict["cumPixels"] += indexToPixel[ blenderToIndex[ obj ] ]
            for dependent in dependents_map[obj]:
                row_dict["cumPixels"] += indexToPixel[ blenderToIndex[ dependent ] ]
        row_dict["cumPixelsPercent"] = round( row_dict["cumPixels"] / totalPixels , 4 )

        # unpack all the tensors with .item() and round to 4 dec places
        row_dict["modelGroundProb"] = round( row_dict["modelGroundProb"].item(), 4 )
        row_dict["modelOrigProb"] = round( row_dict["modelOrigProb"].item(), 4 )
        row_dict["normGroundProb"] = round( row_dict["normGroundProb"].item(), 4 )
        row_dict["normOrigProb"] = round( row_dict["normOrigProb"].item(), 4 )
        row_dict["normOrigIdx"] = row_dict["normOrigIdx"].item() # Don't round the ID!
        row_dict["deltaGroundProb"] = round( row_dict["deltaGroundProb"].item(), 4 )
        row_dict["deltaOrigProb"] = round( row_dict["deltaOrigProb"].item(), 4 )

        data_table.append(row_dict)
#----------------------------------------------------------------------



#----------------------------------------------------------------------
if __name__=="__main__":
    output_folder = "/n/fs/obj-cv/experiment_project/savedRemovals/diningroom99" # for the data set of transformed images
    dataset_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/diningRoom99" # original dataset
    data_write_folder = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/dataFolder/diningroom97Data" # "output" folder for where all the data gets written to

    # placesModel
    CNN = placesCNN()

    # hard coded ground truth
    # ground_id = 203 # kitchen
    # ground_id = 45 # bathroom
    # ground_id = 215 # livingroom
    ground_id = 121 # diningroom
    # ground_id = 52 # bedroom

    # Comment this out for no arguments
    parser = argparse.ArgumentParser(description = "analyzeDataParams")
    parser.add_argument("--output_folder", required = True, help="output_folder")
    parser.add_argument("--dataset_path", required = True, help="dataset_path")
    parser.add_argument("--data_write_folder", required = True, help="data_write_folder")
    parser.add_argument("--ground_id", required = True, help="ground_id")
    args = parser.parse_args()

    output_folder = args.output_folder
    dataset_path = args.dataset_path
    data_write_folder = args.data_write_folder
    output_folder = args.output_folder
    ground_id = int(args.ground_id)

    ground_name = CNN.classes[ground_id]

    # data_table. Building a data table that we can put into pandas later
    data_table = []

    # Option 1: Just do once scene COMMENT OUT ONE OF THE OPTIONS
    # scene_id = "1e122544" # current scene we on
    # analyze_scene(output_folder, dataset_path, scene_id, CNN, data_table)

    # Option 2: Do every scene in output_folder
    # For each scene_id in the output_folder, add it's rows to the scene_dataframe
    total_scenes = len(os.listdir(output_folder))
    skipped_count = 0
    for index, scene_id in enumerate( sorted( os.listdir(output_folder)) ):
        if scene_id=="logs" or scene_id=="infoFolder": # don't want these two folders
            skipped_count += 1
            continue
        if not os.path.isdir(os.path.join(output_folder, scene_id)):
            print("How did a loose file get in the output_folder. exiting...", file=sys.stderr)
        # if you want to break early
        # if index==2:
        #     print("Early Break Scheduled", file=sys.stderr)
        #     break
        analyze_scene(output_folder, dataset_path, scene_id, CNN, data_table) # updates data_table
        print(f"Progress: {index + 1}/{total_scenes} \t Skipped: {skipped_count}", flush=True)

    #***********************
    # Only work in pandas once data_table is completed
    #***********************
    df = pd.DataFrame(data_table)
    df = df.assign( groundId=ground_id )
    df = df.assign( groundName=ground_name)
    with open(os.path.join(data_write_folder, "rawData.csv"), "w") as file:
        file.write(df.to_csv())
    with open(os.path.join(data_write_folder, "rawData.txt"), "w") as file:
        file.write(df.to_string())


    # Create Cleaned Data for single objects
    singleRows = df[ df["isSingle"] ]
    single_df = (
        singleRows.groupby(
            "groupName"
        ).agg(
            avgNormGroundProb = ('normGroundProb', 'mean'),
            avgNormOrigProb = ('normOrigProb', 'mean'),
            avgDeltaGroundProb = ('deltaGroundProb', 'mean'),
            avgDeltaOrigProb = ('deltaOrigProb', 'mean'),
            cumPixelsPercent = ('cumPixelsPercent', 'mean'),
            numScenesPresent = ('sceneId', 'nunique')
        )
    )
    single_df = single_df.round( {"avgDeltaGroundProb": 4, "avgDeltaOrigProb": 4, "cumPixelsPercent": 4} )

    with open(os.path.join(data_write_folder, "single.csv"), "w") as file:
        file.write(single_df.to_csv())
    with open(os.path.join(data_write_folder, "single.txt"), "w") as file:
        file.write(single_df.to_string())


    # Create cleaned data for multi (all) removal objects
    multiRows = df[ df["isAll"] ]
    multi_df = (
        multiRows.groupby(
            "groupName"
        ).agg(
            avgNormGroundProb = ('normGroundProb', 'mean'),
            avgNormOrigProb = ('normOrigProb', 'mean'),
            avgDeltaGroundProb = ('deltaGroundProb', 'mean'),
            avgDeltaOrigProb = ('deltaOrigProb', 'mean'),
            cumPixelsPercent = ('cumPixelsPercent', 'mean'),
            numScenesPresent = ('sceneId', 'nunique')
        )
    )
    multi_df = multi_df.round( {"avgDeltaGroundProb": 4, "avgDeltaOrigProb": 4, "cumPixelsPercent": 4} )

    with open(os.path.join(data_write_folder, "multi.csv"), "w") as file:
        file.write(multi_df.to_csv())
    with open(os.path.join(data_write_folder, "multi.txt"), "w") as file:
        file.write(multi_df.to_string())


    merged_df = pd.concat( [single_df.assign(isSingle=True), multi_df.assign(isSingle=False)], axis=0 )
    cols = [ 'isSingle' ] + [ col for col in merged_df.columns if col != 'isSingle' ]
    merged_df = merged_df[ cols ]
    merged_df = merged_df.sort_values(by="avgDeltaGroundProb")
    with open(os.path.join(data_write_folder, "merged.csv"), "w") as file:
        file.write(merged_df.to_csv())
    with open(os.path.join(data_write_folder, "merged.txt"), "w") as file:
        file.write(merged_df.to_string())

#----------------------------------------------------------------------
