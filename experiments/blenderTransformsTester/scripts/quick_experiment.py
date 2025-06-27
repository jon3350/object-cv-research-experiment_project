import bpy, yaml, os, json
from transforms.camera import zoom_camera_set, zoom_camera_get_baseline, rotate_camera_get_baseline, rotate_camera_set
from transforms.object import slide_object, rotate_object, remove_object, restore_object, add_arrow
from mathutils import Vector

def render_and_log(render_file_path):
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still = True)

def apply_scene(scene_cfg):
    base_path = scene_cfg["base_path"]
    results_path = scene_cfg["results_path"]
    # open blender scene
    bpy.ops.wm.open_mainfile(filepath = os.path.join(base_path, "scene.blend"))
    # configure GPU
    config_GPU_and_render_quality()

    with open( os.path.join(base_path, "solve_state.json"), "r") as f:
        solve_state = json.load(f)

    # get active camera
    cam = bpy.context.scene.camera.name

    # Object Deletion
    chosen_objs = scene_cfg["objects"]["pick"]

    # add_arrow(Vector((0,0,0)), Vector((0,0,1)))
    cam = bpy.context.scene.camera
    table_blender_name = "TableDiningFactory(8071207).spawn_asset(3272403)"
    print(bpy.data.objects[table_blender_name].location)
    add_arrow( Vector((4.7630, 5.2811, 0.1575)), Vector((4.7630, 5.2811, 1.1575)) )
    add_arrow( Vector((4.7630, 5.2811, 0.1575)), Vector((4.7630, 6.2811, 0.1575)) )
    add_arrow( Vector((4.7630, 5.2811, 0.1575)), Vector((5.7630, 5.2811, 0.1575)) )
    # oldLoc, newLoc, axisRet, movementVector = slide_object(table_blender_name, (0.9999894101926659, -1.9700374390311905e-05, 0.003254121553565114), 1 )
    # oldLoc, newLoc, axisRet, movementVector = slide_object(table_blender_name, (-1.9700374390311905e-05, 0.9999633511036724, 0.0060536902036071654), 1 )
    oldLoc, newLoc, axisRet, movementVector = slide_object(table_blender_name, (0.003254121553565114, 0.0060536902036071654, 4.723870366152294e-05), 1 )
    render_and_log( os.path.join(results_path, "vector1" ))
    print(oldLoc, newLoc, axisRet, movementVector)

    # Object Sliding
    # slide_min, slide_max, slide_step = scene_cfg["objects"]["slide"]
    # for obj_key in chosen_objs:
    #     meta = solve_state["objs"][obj_key]
    #     blender_name = meta["obj"]

    #     # read the DOF matrix
    #     M = meta.get("dof_matrix_translation")
    #     if not M:
    #         print(f"{obj_key} has no translation DOF, skipping")
    #         continue
        
    #     # record original location
    #     obj = bpy.data.objects[blender_name]
    #     baseline_loc = obj.location.copy()

    #     # for i in range(len(M)):
    #     for i in range(1):
    #         # extract column vector
    #         axis = (M[0][i], M[1][i], M[2][i])

    #         # skip zero axis
    #         if abs(axis[0]) < 1e-3 and abs(axis[1]) < 1e-3 and abs(axis[2]) < 1e-3:
    #             continue
            
    #         for dist in range(int(slide_min*10), int(slide_max*10) + 1, int(slide_step*10)):
    #             dist /= 10
    #             slide_object(blender_name, axis, dist)
    #             render_and_log( os.path.join(results_path, f"obj_slide_{obj_key}_dof{i}_{dist:+.2f}.png") )
    #             obj.location = baseline_loc.copy()

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
    bpy.context.scene.cycles.samples            = 512
    bpy.context.scene.render.resolution_percentage = 50

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
    cfg = yaml.safe_load(open("config/experiment.yaml"))
    for scene_cfg in cfg["scenes"]:
        apply_scene(scene_cfg)