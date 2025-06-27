import bpy
from math import radians
from mathutils import Vector, Quaternion

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

def slide_object(obj_name: str, axis, distance: float):
    """
    Axis should be 3 tuple
    """
    obj = bpy.data.objects[obj_name]
    vec = Vector(axis)

    oldLoc = obj.location.copy()
    axis = vec.copy()

    # Normalize it so distance always means “that many meters”
    if vec.length == 0:
        raise ValueError("Axis vector cannot be zero")
    vec.normalize()

    movementVector = vec * distance
    obj.location += vec * distance
    newLoc = obj.location.copy()

    return (oldLoc, newLoc, axis, movementVector)


def rotate_object(obj_name: str, axis, angle_deg: float):
    obj = bpy.data.objects[obj_name]
    q = Quaternion(axis, radians(angle_deg) )
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = q @ obj.rotation_quaternion


def remove_object(obj_name: str, remove_children: bool = False, solve_state: dict = None):
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

def restore_object(old_flags):
    for o, flag in old_flags.items():
        o.hide_render = flag