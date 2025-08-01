# general imports
import pandas as pd
import os
import argparse

# imports for blender staples
from utils.blenderStaples import render_and_log, config_GPU_and_render_quality

# imports for master dict

# imports for placesCNN()

# imports for move_on_top
import bpy
from utils.mini_solver import *


#----------------------------------------------------------------------
if __name__=="__main__":
    # Kitchen
    output_folder = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/results" # for the data set of transformed images
    dataset_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/bedroom96" # original dataset
    index = 109

    # ***Use parser if using parallelization
    # parser = argparse.ArgumentParser(description = "swapObjects.py")
    # parser.add_argument("--output_folder", required = True, help="output_folder")
    # parser.add_argument("--dataset_path", required = True, help="dataset_path")
    # parser.add_argument("--index", required = True, help="index")
    # args = parser.parse_args()

    # output_folder = args.output_folder
    # dataset_path = args.dataset_path
    # index = int(args.index)

    # read the pd as csv
    df = pd.read_csv( "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/filteredBedroomSwap.csv" )

    # temporary fix since couldn't get ahold of gpus
    for index in range( 0, 4 ):

        # for each row, extract the dest and target
        row = df.loc[index]
        target, dest = row["target"], row["dest"]
        scene_id = row["sceneId"]

        # open blender scene
        blender_scene_path = os.path.join( dataset_path, scene_id, "fine", "scene.blend" )
        bpy.ops.wm.open_mainfile(filepath = blender_scene_path)
        print(flush=True)
        config_GPU_and_render_quality( samples=1024, resolution_percentage=50)
        # config_GPU_and_render_quality( samples=8192, resolution_percentage=100)

        # call custom functions to move small object on big object************
        small_obj = bpy.data.objects[target]
        big_obj = bpy.data.objects[dest]

        # orient the small_obj
        align_small_object_with_parenting( big_obj, small_obj )
    
        # SPECIAL: BLANKET
        # Just lazily place above big_obj's bound box and then run cloth simulation for 30 frames
        if "Blanket" in small_obj.name or "Comforter" in small_obj.name:
            put_cloth_in_best_loc( small_obj, big_obj, bpy.data.objects["bedroom_0/0.floor"] )
        
        # for lamps just use naive aligning
        elif "Lamp" in big_obj.name:
            put_middle_on_top( small_obj, big_obj )
            
        # use ray tracing for everything else
        else:

            # get big_obj bound_box
            big_mmD = max_min_bound_box( big_obj, local=True )

            # SPECIAL: Update kitchenSpace upper bound box to only include bottom counter
            if "KitchenSpace" in big_obj.name:
                max_gap, max_pair = largest_z_gap( big_obj )
                big_mmD["z_max"] = ( max_pair[0] + max_pair[1] ) / 2

            # raycast to get hit vertices
            hits = raycast_top_grid( big_obj, big_mmD["x_min"], big_mmD["x_max"], 
                                    big_mmD["y_min"], big_mmD["y_max"], big_mmD["z_max"],
                                    local_input= True, local_output = True )
                
                                    
            # convert collection of vectices to a plane (5-tuple)
            top_plane = find_top_plane( hits ) # top plane is in local coordinates right now
            top_plane = ( big_obj.matrix_world @ top_plane[0], 
                        big_obj.matrix_world @ top_plane[1],
                        big_obj.matrix_world @ top_plane[2],
                        big_obj.matrix_world @ top_plane[3] )
            print("TOPPLANE", top_plane)
            
            
            # SPECIAL: For convex objects, you are allowed to go higher than the top plane
            # the top plane is often the bottom floor
            if "Bathtub" in big_obj.name or "StandingSink" in big_obj.name:
                z_arr = [] # store each partitions ( info_matrix, best_tuple )
                z_paritions = 10
                global_big_mmD = max_min_bound_box( big_obj, local=False )
                orig_z = top_plane[0].z 
                delta_z = global_big_mmD["z_max"] - top_plane[0].z
                for i in range(z_paritions+1):
                    top_plane[0].z = orig_z + ( i/z_paritions ) * delta_z
                    top_plane[1].z = orig_z + ( i/z_paritions ) * delta_z
                    top_plane[2].z = orig_z + ( i/z_paritions ) * delta_z
                    top_plane[3].z = orig_z + ( i/z_paritions ) * delta_z
                    
                    info_matrix = solve_xy_on_flat_surface( small_obj, big_obj, top_plane, dataset_path, scene_id, check_big_intersect = True )
                    score_list, best_tuple = put_object_in_best_loc( small_obj, info_matrix )
                    # put a tax of up to -750 based on object height
                    best_tuple = ( best_tuple[0], best_tuple[1], best_tuple[2] - 750 * ( i/z_paritions ) )
                    z_arr.append( (info_matrix, best_tuple) )
                
                # find the best socre across all z-levels and place object there
                best_in_z_arr = max( z_arr, key = lambda x: x[1][2] )
                put_object_in_best_loc( small_obj, best_in_z_arr[0] )
                    
                    
            else:
                # get info maxtrix using solver
                info_matrix = solve_xy_on_flat_surface( small_obj, big_obj, top_plane, dataset_path, scene_id )

                # using info matrix place small object in the best location
                score_list, best_tuple = put_object_in_best_loc( small_obj, info_matrix )
                #*******************************end of moving object to best place

        
        # name image after index # scene_id # target_object_name
        image_name = f"{index}#{scene_id}#{target}"
        render_and_log(os.path.join(output_folder, image_name ))

#----------------------------------------------------------------------
