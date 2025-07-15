# In objRemovalExp/removeObjectsParams not many functions here are actually used
# Really only just remove_object_blender_name

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import bmesh

def add_arrow(start: Vector,
              end:   Vector,
              shaft_radius: float = 0.01,
              head_radius:  float = 0.02,
              head_length:  float = 0.1):
    """
    Create an arrow mesh pointing from `start` to `end`.
    - shaft: cylinder of radius `shaft_radius`
    - head : cone of radius `head_radius` and length `head_length`
    """
    # direction & length
    vec = end - start
    length = vec.length
    if length < 1e-6:
        return None
    dir_n = vec.normalized()

    # ─── Shaft ──────────────────────────────────────────────────────────
    # place cylinder so its center is halfway along (minus head)
    shaft_len = length - head_length
    shaft_loc = start + dir_n * (shaft_len/2)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius,
        depth=shaft_len,
        location=shaft_loc
    )
    shaft = bpy.context.object
    shaft.rotation_mode = 'QUATERNION'
    # align cylinder's Z axis to the arrow direction
    shaft.rotation_quaternion = dir_n.to_track_quat('Z','Y')
    shaft.name = "Arrow_Shaft"

    # ─── Head ────────────────────────────────────────────────────────────
    head_loc = start + dir_n * (shaft_len + head_length/2)
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_radius,
        depth=head_length,
        location=head_loc
    )
    head = bpy.context.object
    head.rotation_mode = 'QUATERNION'
    head.rotation_quaternion = dir_n.to_track_quat('Z','Y')
    head.name = "Arrow_Head"

    return (shaft, head)

def show_bounding_box(obj_name, solve_state: dict):
    # pick your object
    blender_obj_name = solve_state["objs"][obj_name]["obj"]
    # obj = bpy.context.scene.objects[blender_obj_name]
    obj = bpy.data.objects[blender_obj_name]

    # grab its world‐space corners
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    verts = [(v.x, v.y, v.z) for v in corners]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    # make a new mesh for the bbox
    mesh = bpy.data.meshes.new(obj.name + "_BBox")
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    # turn it into an object and link it to the scene
    bbox_obj = bpy.data.objects.new(obj.name + "_BBox", mesh)
    bpy.context.collection.objects.link(bbox_obj)

    # give it a simple wireframe material (optional)
    mat = bpy.data.materials.new("BBoxMat")
    mat.diffuse_color = (1,0,0,1)
    mat.blend_method = 'BLEND'
    mesh.materials.append(mat)
    bbox_obj.show_wire = True
    bbox_obj.show_all_edges = True
    # added stufffffffffff
    mod = bbox_obj.modifiers.new(name="BBoxWire", type='WIREFRAME')
    mod.thickness = 0.02
    mod.use_replace = True

def get_objects_in_camera_frame(solve_state: dict):
    obj_list = []
    # for key, meta in solve_state["objs"].items():
    #     if corners_in_camera_frame(key, solve_state) > 0:
    #         obj_list.append(key)
    # return obj_list

    for key, meta in solve_state["objs"].items():
        blender_obj = bpy.data.objects[ solve_state["objs"][key]["obj"] ]
        vert_inside_count, total_verts = count_vertices(blender_obj)
        if vert_inside_count / total_verts > 0.2:
            obj_list.append(key)
    return obj_list


def corners_in_camera_frame(obj_name: str, solve_state: dict):
    """
    Takes the solve_state obj name and the solve_state dict
    Returns the number of the object's corners that are in the camera's frame
    (could be blocked by other objects)
    """
    scene = bpy.context.scene
    cam = scene.camera

    # get the blender object
    blender_obj_name = solve_state["objs"][obj_name]["obj"]
    blender_obj = scene.objects[blender_obj_name]
    
    num_corners_inside = 0
    # go through each corner
    for corner in blender_obj.bound_box:
        # convert corner from local to world cordiantes
        cord_world = blender_obj.matrix_world @ Vector(corner)

        # project into camera view (normalized device coordinates)
        cord_ndc = world_to_camera_view(scene, cam, cord_world)

        # check if inside frustum
        # print(Vector(cord_ndc))
        if 0 <= cord_ndc.x <= 1 and 0 <= cord_ndc.y <= 1 and 0 <= cord_ndc.z:
            num_corners_inside += 1

    return num_corners_inside            


def remove_object(obj_name: str, remove_children: bool = True, solve_state: dict = None):
    """
    TAKES THE JSON OBJ NAME NOT THE BLENDER ONE
    Hide (remove) obj_name, and—if remove_children=True—also every object
    that has a StableAgainst relation pointing at obj_name (recursively).
    solve_state should be the dict loaded from solve_state.json.
    Returns a dict mapping each hidden object → its previous hide_render flag.
    """
    # Build a set of Blender object names to hide
    names_to_hide = {obj_name}
    if remove_children:
        # do a BFS over the scene-graph
        queue = [obj_name]
        while queue:
            parent = queue.pop(0)
            # For each object in solve_state, see if it lists parent as a target
            for key, meta in solve_state.get("objs", {}).items():
                for rel in meta.get("relations", []):
                    relation_type = rel.get("relation", {}).get("relation_type")
                    target_name = rel.get("target_name")
                    if relation_type:
                        child_tags = rel["relation"].get("child_tags", [])
                    # print(relation_type, target_name, parent)
                    if relation_type == "StableAgainst" and target_name == parent and "Subpart(bottom)" in child_tags:
                        # the child name is the solve_state json name
                        names_to_hide.add(key)
                        queue.append(key)

    
    # Lookup the actual bpy.data.objects
    objs = []
    for solve_state_json_name in names_to_hide:
        blender_obj_name = solve_state["objs"][solve_state_json_name]["obj"]
        o = bpy.data.objects[blender_obj_name]
        if o:
            objs.append(o)
    
    # record old flags
    old_flags = {o: o.hide_render for o in objs}

    # hide them
    for o in objs:
        o.hide_render = True
    
    # return the flags
    return old_flags

def remove_object_blender_name(obj_name: str):
    """
    Returns a dict mapping each hidden object → its previous hide_render flag.
    """    
    # record old flags
    old_flags = {o: o.hide_render for o in bpy.data.objects}

    # hide them
    bpy.data.objects[obj_name].hide_render = True
    
    # return the flags
    return old_flags

def restore_object(old_flags):
    for o, flag in old_flags.items():
        o.hide_render = flag

def count_vertices(obj):
    scene = bpy.context.scene
    cam = scene.camera
    mesh = obj.data
    mat_world = obj.matrix_world
    cs, ce = cam.data.clip_start, cam.data.clip_end
    # print(cs, ce)
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    
    vert_inside_count = 0
    for v in bm.verts:
        co_ndc = world_to_camera_view(scene, cam, mat_world @ v.co)
        # print(co_ndc)
        # check whether point is inside frustum
        if (0.0 < co_ndc.x < 1.0 and
            0.0 < co_ndc.y < 1.0 and
            cs < co_ndc.z < ce):
            # v.select = True
            vert_inside_count += 1
        else:
            # v.select = False
            pass
    total_verts = len(bm.verts)
    return (vert_inside_count, total_verts )


def create_cube_from_vertices(verts, name="CustomCube"):
    """
    Create a mesh object whose corners are the 8 vectors in `verts`.
    verts: list of 8 (x, y, z) tuples or Vector objects, in any consistent ordering.
    """
    # Define the 6 faces by indexing into verts.
    # This assumes your verts are ordered like a typical cube:
    #   0:(-x,-y,-z), 1:(+x,-y,-z), 2:(+x,+y,-z), 3:(-x,+y,-z),
    #   4:(-x,-y,+z), 5:(+x,-y,+z), 6:(+x,+y,+z), 7:(-x,+y,+z)
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front
        (1, 2, 6, 5),  # right
        (2, 3, 7, 6),  # back
        (3, 0, 4, 7),  # left
    ]

    # Create a new mesh and fill in the verts/faces
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Create an object with that mesh, link into the scene
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def create_bounding_box_cube(obj):
    world_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bbox_cube = create_cube_from_vertices(world_corners, name="idk")
    bbox_cube.display_type = 'SOLID'
    bbox_cube.show_in_front = True
    return bbox_cube

def delete_object_direct(obj):
    # Unlink and remove the object (and its data-block if unused)
    bpy.data.objects.remove(obj, do_unlink=True)