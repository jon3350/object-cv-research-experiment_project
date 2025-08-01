# mini solver
"""
Mini Solver
"""

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import bmesh
from mathutils.bvhtree import BVHTree

from .getMasterDict import getMasterDict

#----------------------------------------------------------------------
# CUSTOM FUNCTIONS

# GENERAL HELPER
def max_min_bound_box( obj, local = False ):
    for idx, corner in enumerate(obj.bound_box):
        if local:
            vec_corner = Vector(corner) # coordinates relative to object origin (obj.location)
        else:
            vec_corner = obj.matrix_world @ Vector(corner)
        if idx == 0:
            max_min_dict = {
                "x_max": vec_corner.x,
                "x_min": vec_corner.x,
                "y_max": vec_corner.y,
                "y_min": vec_corner.y,
                "z_max": vec_corner.z,
                "z_min": vec_corner.z
            }
        else:
            max_min_dict["x_max"] = max( max_min_dict["x_max"], vec_corner.x )
            max_min_dict["x_min"] = min( max_min_dict["x_min"], vec_corner.x )
            max_min_dict["y_max"] = max( max_min_dict["y_max"], vec_corner.y )
            max_min_dict["y_min"] = min( max_min_dict["y_min"], vec_corner.y )
            max_min_dict["z_max"] = max( max_min_dict["z_max"], vec_corner.z )
            max_min_dict["z_min"] = min( max_min_dict["z_min"], vec_corner.z )
    return max_min_dict

# V1 HELPERS
# note: SOLVER_XY V1 NOT INCLUDED
def corners_in_clip(scene, cam_obj, obj):
    cam_data = cam_obj.data
    mat = obj.matrix_world
    
    # count how many corners are in the clip
    count = 0

    for corner in obj.bound_box:
        co_world = mat @ Vector(corner)
        ndc = world_to_camera_view(scene, cam_obj, co_world)
        # ndc.x/y in [0,1] is in‐frame; ndc.z is the depth on the view axis
        if (0.0 <= ndc.x <= 1.0 and
            0.0 <= ndc.y <= 1.0 and
            cam_data.clip_start < ndc.z < cam_data.clip_end):
            count += 1
    return count


def is_center_visible_by_raycast(scene, cam_obj, target_obj):
    """See if raycast from camera can hit center of object's bound box"""
    cam_loc = cam_obj.matrix_world.to_translation()
    # test against object’s bounding‐box center
    bbox_center = sum((Vector(c) for c in target_obj.bound_box), Vector()) / 8.0
    wc_center = target_obj.matrix_world @ bbox_center
    direction = (wc_center - cam_loc).normalized()

    result, hit_loc, hit_norm, _, hit_obj, _ = scene.ray_cast(
        depsgraph=bpy.context.evaluated_depsgraph_get(),
        origin=cam_loc,
        direction=direction
    )
    return result and hit_obj == target_obj

def count_ray_hits_to_camera(scene, cam_obj, target_obj, epsilon=1e-4):
    """
    Cast rays from the target_obj’s 8 bounding‑box corners
    toward the camera. Return how many of those 8 rays reach the 
    camera without hitting any meshes
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc   = cam_obj.matrix_world.to_translation()

    # build the 9 sample points in local space
    local_corners = [Vector(c) for c in target_obj.bound_box]

    hit_count = 0
    for lp in local_corners:
        # world‑space origin of the ray
        origin    = target_obj.matrix_world @ lp
        direction = (cam_loc - origin).normalized()

        # offset a bit so we don’t immediately self‑hit
        origin_offset = origin + direction * epsilon
        max_dist      = (cam_loc - origin_offset).length

        hit, hit_loc, hit_norm, face_idx, hit_obj, matrix = scene.ray_cast(
            depsgraph,
            origin=origin_offset,
            direction=direction,
            distance=max_dist
        )
        
        # If no mesh is hit then the ray hit the camera safely
        if not hit:
            hit_count += 1

    return hit_count

def intersects_objects(small_obj, big_obj, dataset_path, scene_id, check_big_intersect = False):
    """determine if small_obj intersects any dependents of the big_obj (except the small object itself)"""
    mDct = getMasterDict( dataset_path, scene_id )
    small_obj_name = small_obj.name
    big_obj_name = big_obj.name

    if check_big_intersect and mesh_intersect( small_obj, big_obj ):
        return True

    for dependent in mDct["dependents_map"][big_obj_name]:
        if dependent == small_obj_name: # don't want to check if intersects with self
            continue
        if mesh_intersect( small_obj, bpy.data.objects[dependent] ):
            return True
    return False

def mesh_intersect(obj_a, obj_b):
    """ determine if two objects are intersecting """
    bm_a = bmesh.new()
    bm_b = bmesh.new()
    bm_a.from_object(obj_a, bpy.context.evaluated_depsgraph_get())
    bm_b.from_object(obj_b, bpy.context.evaluated_depsgraph_get())

    bm_a.transform(obj_a.matrix_world)
    bm_b.transform(obj_b.matrix_world)

    tree_a = BVHTree.FromBMesh(bm_a)
    tree_b = BVHTree.FromBMesh(bm_b)

    intersect = tree_a.overlap(tree_b)

    bm_a.free()
    bm_b.free()
    return len(intersect) > 0


# V2 FUNCTIONS
def raycast_top_grid( obj, x_min, x_max, y_min, y_max, z_origin, local_input = False, local_output = False, partitions=100 ):
    """ castsrays from a grid defined by x_min, x_max, y_min, y_max, z_origin, partitions
     reutrns an array of all vertices of obj that were hit 
     use local_input and local_output to determine if the input coordinates and return values should be in local or global space
     Z is assumed to be down for both local and global space for the purposes of raycast direction
     """
    hits = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    direction = Vector((0,0,-1))

    # if using local input coordinates, transform direction ray cast
    # (should be the same since we assume origin is z-aligned)
    if local_input:
        direction = obj.matrix_world.to_3x3() @ direction
        direction.normalize()

    for i in range(partitions):
        for j in range(partitions):
            x_origin = x_min + (x_max - x_min) * (i/(partitions-1))
            y_origin = y_min + (y_max - y_min) * (j/(partitions-1))
            origin = Vector( (x_origin, y_origin, z_origin) )

            # if using local coordinates, convert ray cast origin to global coordintes
            if local_input:
                origin = obj.matrix_world @ origin

            result, hit_loc, hit_norm, hit_idx, hit_obj, hit_mat = scene.ray_cast(
                depsgraph, origin + direction*1e-3, direction
            )
            if result:
                hit_vertex = hit_loc
                # if using local coordinates convert global hit_vertex to local coordinates
                if local_output:
                    hit_vertex = obj.matrix_world.inverted() @ hit_vertex

                hits.append( hit_vertex )

    return hits


def find_top_plane( hits ):
    """ takes the hits array from raycast_top_grid and returns a flat plane using
    the top_left, top_right, bot_right, bot_left vectors"""

    # if somehow all the rays missed and hits is empty
    if not hits:
        raise ValueError("No ray hits—can't compute top plane.")

    hits_x = sorted(hits, key=lambda vec: vec.x)
    hits_y = sorted(hits, key=lambda vec: vec.y)
    hits_z = sorted(hits, key=lambda vec: vec.z)
    z_median = hits_z[ len(hits)//2 ].z

    # filter out points that are more than epsilon away from z-median
    eps = 1e-3
    hits_x = [ point for point in hits_x if abs(point.z - z_median) < eps ]
    hits_y = [ point for point in hits_y if abs(point.z - z_median) < eps ]

    x_min = hits_x[0].x
    x_max = hits_x[len(hits_x)-1].x
    y_min = hits_y[0].y
    y_max = hits_y[len(hits_y)-1].y

    top_left = Vector( (x_min, y_max, z_median) )
    top_right = Vector( (x_max, y_max, z_median) )
    bot_right = Vector( (x_max, y_min, z_median) )
    bot_left = Vector( (x_min, y_min, z_median) )

    return ( top_left, top_right, bot_right, bot_left )


def solve_xy_on_flat_surface( small_obj, big_obj, top_plane, dataset_path, scene_id, partitions=10, check_big_intersect = False ):
    """Given a top_plane, align small_obj's bottom z to top_plane's z. Then slide small_obj's xy coordinates
    along the plane.
    Returns 2D info matrixs where each element is tuple
    ( corners_in_clip, is_center_visible_by_raycast, count_ray_hits_to_camera, intersects_objects ) 
    
    [0, partitions] is the interval used to slide objects around in the x and y axis
    
    top_plane should be in the form ( x_min, x_max, y_min, y_max, z_median )
    which is what find_top_plane returns

    Work in global coordinates only
    
    Check Big Intersect - if set to true then checks the big object for intersection. Used for convex objects like sinks
    """

    plane_tl, plane_tr, plane_br, plane_bl = top_plane

    # have the middle of the small_obj align with the top left corner of top_plane
    small_obj_mmD = max_min_bound_box(small_obj)
    small_obj_cx = (small_obj_mmD["x_min"] + small_obj_mmD["x_max"]) / 2 # small object center
    small_obj_cy = (small_obj_mmD["y_min"] + small_obj_mmD["y_max"]) / 2
    center_to_top_left_corner = Vector(( plane_tl.x - small_obj_cx, plane_tl.y - small_obj_cy, 
                                        plane_tl.z - small_obj_mmD["z_min"] ))
    
    # Compute your “base” matrix once
    base_mw = small_obj.matrix_world.copy()
    base_mw.translation += center_to_top_left_corner
    # Move object to the top–left only once
    small_obj.matrix_world = base_mw
    bpy.context.view_layer.update()

    dir_lr = plane_tr - plane_tl # left to right direction
    dir_bt = plane_bl - plane_tl # top to bottom direction

    # preallocate your score matrix
    info_matrix = [[None]*(partitions+1) for _ in range(partitions+1)]

    for i in range(partitions + 1):
        for j in range(partitions + 1):
            mw = base_mw.copy()
            mw.translation += (i/partitions) * dir_lr + (j/partitions) * dir_bt
            small_obj.matrix_world = mw
            bpy.context.view_layer.update()

            # update info matrix:
            scene = bpy.context.scene
            cam = scene.camera
            info_matrix[j][i] = (corners_in_clip(scene, cam, small_obj), 
                                 is_center_visible_by_raycast(scene, cam, small_obj),
                                 count_ray_hits_to_camera(scene, cam, small_obj),
                                intersects_objects(small_obj, big_obj, dataset_path, scene_id, check_big_intersect = check_big_intersect),
                                 mw )
            
            # just for testing we render as well
            # import os
            # from .blenderStaples import render_and_log
            # print(i, j)
            # if i%4 == 0 and j % 4 == 0:
            #     img_name = f"temp_img(j={j}, i={i} )"
            #     render_and_log( os.path.join( "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp", img_name ) )

    return info_matrix


def align_small_object_with_parenting( parent, child ):
    """ Used to have child's axis line up with parent's axis
     Front is +X in infinigen """
    child.parent = parent

    # zero out child transforms
    child.matrix_basis.identity()
    
    # — Default: “no inverse” → child inherits parent’s transform immediately
    child.matrix_parent_inverse.identity()

    bpy.context.view_layer.update()


# ADD ON: KITCHEN SPACE
# specific add on for kitchenSpace
def largest_z_gap(obj):
    # 1) Collect all the Z‐coordinates
    zs = []
    
    # transform every 50 local vertex into world space:
    for i in range( 0, len(obj.data.vertices), 50 ):
        v = obj.data.vertices[i]
        co_world = obj.matrix_world @ v.co
        zs.append(co_world.z)

    # 2) Sort them
    zs_sorted = sorted(zs)

    # 3) Walk the sorted list and track the maximum gap
    max_gap = 0.0
    max_pair = (None, None)
    for a, b in zip(zs_sorted, zs_sorted[1:]):
        gap = b - a
        if gap > max_gap:
            max_gap = gap
            max_pair = (a, b)

    return max_gap, max_pair


# ADD ON: BEST LOCATION
# add on to place object in best location

# info_matrix[j][i] = (corners_in_clip(scene, cam, small_obj), 
#                      is_center_visible_by_raycast(scene, cam, small_obj),
#                      count_ray_hits_to_camera(scene, cam, small_obj),
#                     intersects_objects(small_obj, big_obj, dataset_path, scene_id),
#                      mw )
def put_object_in_best_loc( obj, info_matrix ):
    """ Given an info matrix for an obj, move the obj to the 'best' location """
    score_list = [] # element = ( i, j, score )

    n = len( info_matrix )
    for i in range( n ):
        for j in range( n ):
            score = 0
            info = info_matrix[j][i] # info matrix is index x,y or j,i

            # if nothing is in clip, instant -1000pts
            if info[0] == 0:
                score_list.append( (i,j, -1000) )
                continue
            else:
                # ***1000pts if all vertex in clip
                score += 1000 * (info[0] / 8)
            
            # ***500pts if center is visible
            if info[1]:
                score += 500
            
            # ***750pts for not intersecting other objects 
            if not info[3]:
                score += 750
            
            # ***500pts for corners being visible (front of object does block back corners)
            score +=  500 * (info[2] / 8)

            # edge case: center and all corners not visible, instant -1000
            if not info[1] and info[2]==0:
                score_list.append( (i,j, -1000) )
                continue
            
            # ***-1000pts for being far away from center
            units_from_center = (i- n/2)**2 + (j- n/2)**2
            max_units_from_center = 2 * ( n/2 )**2
            score -= (units_from_center / max_units_from_center) * 1000

            # add to score list
            score_list.append( (i, j, score) )
    
    # get the tuple corresponding to the max score and move object there
    best_tuple = max(score_list, key=lambda x: x[2])
    best_i, best_j, best_score = best_tuple
    mw = info_matrix[best_j][best_i][4]
    obj.matrix_world = mw
    bpy.context.view_layer.update()

    # returns scorelist and best_tuple for analysis purposes
    return score_list, best_tuple


# Additional helper function to naively put small object on big object with bounding boxes
def put_middle_on_top( small_obj, big_obj):
    # have the middle of the small_obj bottom face align with the middle of big_obj's top face
    small_obj_mmD = max_min_bound_box(small_obj)
    big_obj_mmD = max_min_bound_box(big_obj)
    small_obj_cx = (small_obj_mmD["x_min"] + small_obj_mmD["x_max"]) / 2 # small object center
    small_obj_cy = (small_obj_mmD["y_min"] + small_obj_mmD["y_max"]) / 2
    
    # use ray casting for big object center
    big_obj_cx = (big_obj_mmD["x_min"] + big_obj_mmD["x_max"]) / 2 # big object center
    big_obj_cy = (big_obj_mmD["y_min"] + big_obj_mmD["y_max"]) / 2
    
    # use raycasting to find z plane for the big object (i.e. bed)
#    big_mmD = max_min_bound_box(big_obj, local = True) # this one uses local coordinates
#    hits = raycast_top_grid( big_obj, big_mmD["x_min"], big_mmD["x_max"], 
#                            big_mmD["y_min"], big_mmD["y_max"], big_mmD["z_max"],
#                             local_input= True, local_output = True )
#    # convert collection of vectices to a plane (5-tuple)
#    top_plane = find_top_plane( hits ) # top plane is in local coordinates right now
#    top_plane_z_max = (big_obj.matrix_world @ top_plane[0]).z
                     
    small_obj_center = Vector(( small_obj_cx, small_obj_cy, small_obj_mmD["z_min"] ))
    big_obj_center = Vector(( big_obj_cx, big_obj_cy, big_obj_mmD["z_max"] ))
#    print("HI")
#    print(top_plane_z_max)
#    print(big_obj_mmD["z_max"])

    base_mw = small_obj.matrix_world.copy()
    base_mw.translation += ( big_obj_center - small_obj_center )
    small_obj.matrix_world = base_mw
    bpy.context.view_layer.update()


# Additional helper function to create a plane
def create_plane_from_verts(name, verts):
    """
    Create a mesh object named `name` with a single face defined by `verts`.

    Parameters:
    - name (str): the new object's name.
    - verts (list of tuple[float, float, float]): four corner coordinates in order.
    """
    # 1) Create a new mesh and object
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj  = bpy.data.objects.new(name, mesh)

    # 2) Link object into current collection
    bpy.context.collection.objects.link(obj)

    # 3) Build the geometry: verts, no edges, one face (0–1–2–3)
    mesh.from_pydata([Vector(v) for v in verts], [], [(0, 1, 2, 3)])
    mesh.update(calc_edges=True)

    return obj


# Additional helper function to create a rectangular prism
def create_cube_from_verts(name, verts):
    """
    Create a mesh object named `name` with 8 verts and 6 quad faces.

    Parameters:
      name (str): name for new object
      verts (list of tuple): eight (x,y,z) corner coordinates in the order:
          bottom face: 0,1,2,3
          top    face: 4,5,6,7
    """
    # face definitions (each face is a quad of vertex indices)
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 4, 7, 3),  # side
        (1, 5, 6, 2),  # side
        (0, 1, 5, 4),  # side
        (3, 2, 6, 7),  # side
    ]

    # create mesh + object
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj  = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # build geometry
    mesh.from_pydata([Vector(v) for v in verts], [], faces)
    mesh.update(calc_edges=True)

    return obj

# Special: Different object moving function when cloth is the small object
def put_cloth_in_best_loc( small_obj, big_obj, floor):
    """ Put's a blanket like small_obj on top of big_obj and applies cloth simulation
    Gives collision modifer to blanket, big_obj, and floor
    Gives cloth modifer to blanket
    Runs 25 frames
    """
    
    # have the middle of the small_obj bottom face align with the middle of big_obj's top face
    put_middle_on_top( small_obj, big_obj )
    
    # if the big_obj is a lamp then add a small plane to make sure the cloth doesn't fall through
    if "FloorLamp" in big_obj.name:
        # 1) Compute lamp top‐face bbox
        bbox = max_min_bound_box(big_obj)
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        z_top        = bbox["z_max"]

        # 2) Derive cube center + size
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        thickness = 0.20   # adjust if you need thicker collision
        center_z  = z_top - thickness/2

        size_x = (x_max - x_min)
        size_y = (y_max - y_min)

        # 3) Spawn a Blender cube primitive
        bpy.ops.mesh.primitive_cube_add(
            location=(center_x, center_y, center_z)
        )
        lamp_cube = bpy.context.active_object
        lamp_cube.name = "LampCollisionCube"

        # 4) Scale it so its top face exactly matches the lamp’s top
        lamp_cube.scale = (size_x/2, size_y/2, thickness/2)
        bpy.context.view_layer.update()

        # 5) Apply scale so collision uses the correct dimensions
        bpy.ops.object.transform_apply(scale=True)

        # 6) Add and tweak a Collision modifier
        if "Collision" not in lamp_cube.modifiers:
            lamp_cube.modifiers.new("Collision", type='COLLISION')
        col = lamp_cube.modifiers.new("Collision", type='COLLISION')
    
    START_FRAME    = 1
    END_FRAME      = 25

    scene = bpy.context.scene

#    Add Collision modifiers
    if "Collision" not in small_obj.modifiers:
        small_obj.modifiers.new("Collision", type='COLLISION')
    if "Collision" not in big_obj.modifiers:
        big_obj.modifiers.new("Collision", type='COLLISION')
    if "Collision" not in floor.modifiers:
        floor.modifiers.new("Collision", type='COLLISION')

    # Add Cloth modifier to blanket
    if "Cloth" not in small_obj.modifiers:
        cloth_mod = small_obj.modifiers.new("Cloth", type='CLOTH')
    else:
        cloth_mod = small_obj.modifiers["Cloth"]
    
    bpy.context.view_layer.update()

    # Configure Cloth cache to bake exactly frames 1–30
    #    (so it won’t keep extending beyond frame 30)
    pcache = cloth_mod.point_cache
    pcache.frame_start = START_FRAME
    pcache.frame_end   = END_FRAME
    pcache.use_disk_cache = True  # optional, but faster for large sims

    # Set scene frame range
    scene.frame_start = START_FRAME
    scene.frame_end   = END_FRAME

    # Clear any existing bakes and then bake all dynamic caches
    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake=True)

    # Jump to frame 30
    scene.frame_set(END_FRAME)

#----------------------------------------------------------------------


# unit testing
if __name__=="__main__":
    import os
    from .blenderStaples import config_GPU_and_render_quality, render_and_log

    output_folder = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/utils" # for the data set of transformed images
    dataset_path = "/n/fs/obj-cv/infinigen_project/savedDatasets/kitchenDataset94" # original dataset
    scene_id = "2dbefffc"

    target = "LargePlantContainerFactory(949299).spawn_asset(2608831)"
    dest = "KitchenSpaceFactory(3989955).spawn_asset(8637063)"

    # open blender scene
    blender_scene_path = os.path.join( dataset_path, scene_id, "fine", "scene.blend" )
    bpy.ops.wm.open_mainfile(filepath = blender_scene_path)
    print(flush=True)
    config_GPU_and_render_quality()

    # call custom functions to move small object on big object************
    small_obj = bpy.data.objects[target]
    big_obj = bpy.data.objects[dest]

    # orient the small_obj
    align_small_object_with_parenting( big_obj, small_obj )

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

    # get info maxtrix using solver
    info_matrix = solve_xy_on_flat_surface( small_obj, big_obj, top_plane, dataset_path, scene_id )

    # using info matrix place small object in the best location
    score_list, best_tuple = put_object_in_best_loc( small_obj, info_matrix )
    #*******************************end of moving object to best place

    
    # name image after index # scene_id # target_object_name
    image_name = "test_image"
    render_and_log(os.path.join(output_folder, image_name ))