import bpy, yaml, os, json
from transforms.camera import zoom_camera_set, zoom_camera_get_baseline, rotate_camera_get_baseline, rotate_camera_set
from transforms.object import slide_object, rotate_object, remove_object, restore_object


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

    # Zoom the camera
    zoom_min, zoom_max, zoom_delta = scene_cfg["camera"]["zoom"]
    zoom_baseline = zoom_camera_get_baseline(cam)
    for z in range(zoom_min, zoom_max + 1, zoom_delta):
        zoom_camera_set(cam, zoom_baseline, z)
        render_and_log( os.path.join( results_path, f"camera_zoom{z:+03d}.png") )
    zoom_camera_set(cam, zoom_baseline, 0)

    # camera yaw
    yaw_roll_pitch_baseline = rotate_camera_get_baseline(cam)
    yaw_min, yaw_max, yaw_delta = scene_cfg["camera"]["rotate"]["yaw"]
    for yaw in range(yaw_min, yaw_max + 1, yaw_delta):
        rotate_camera_set(camera_name=cam, yaw=yaw, baseline=yaw_roll_pitch_baseline)
        render_and_log( os.path.join(results_path, f"camera_rotate_yaw{yaw:+03d}.png") )
    rotate_camera_set(camera_name=cam, baseline=yaw_roll_pitch_baseline)

    # camera roll
    yaw_roll_pitch_baseline = rotate_camera_get_baseline(cam)
    roll_min, roll_max, roll_delta = scene_cfg["camera"]["rotate"]["roll"]
    for roll in range(roll_min, roll_max + 1, roll_delta):
        rotate_camera_set(camera_name=cam, roll=roll, baseline=yaw_roll_pitch_baseline)
        render_and_log( os.path.join( results_path, f"camera_rotate_roll{roll:+03d}.png") )
    rotate_camera_set(camera_name=cam, baseline=yaw_roll_pitch_baseline)

    # camera pitch
    yaw_roll_pitch_baseline = rotate_camera_get_baseline(cam)
    pitch_min, pitch_max, pitch_delta = scene_cfg["camera"]["rotate"]["pitch"]
    for pitch in range(pitch_min, pitch_max + 1, pitch_delta):
        rotate_camera_set(camera_name=cam, pitch=pitch, baseline=yaw_roll_pitch_baseline)
        render_and_log( os.path.join( results_path, f"camera_rotate_pitch{pitch:+03d}.png") )
    rotate_camera_set(camera_name=cam, baseline=yaw_roll_pitch_baseline)

    # Object Deletion
    chosen_objs = scene_cfg["objects"]["pick"]
    for obj in chosen_objs:
        # obj_name = solve_state["objs"][obj]["obj"] DO NOT USE BLENDER NAME FOR THESE METHODS

        # remove object on it's own
        old_flags = remove_object(obj, solve_state=solve_state)
        render_and_log( os.path.join( results_path, f"remove_obj_alone_{obj}.png") )
        restore_object(old_flags)

        # remove object and stuff on top of it
        old_flags = remove_object(obj, remove_children = True, solve_state=solve_state)
        print(old_flags)
        render_and_log( os.path.join( results_path, f"remove_obj_related_{obj}.png"))
        restore_object(old_flags)

    
    # Object rotation
    ang_min, ang_max, ang_step = scene_cfg["objects"]["rotate"]
    for obj_key in chosen_objs:
        # get blender name and DOF axis
        meta = solve_state["objs"][obj_key]
        blender_name = meta["obj"]
        axis = meta.get("dof_rotation_axis")
        if not axis:
            print(f"obj {obj_key} has no rotation DOF, skipping")
            continue

        # record baseline quaternion
        obj = bpy.data.objects[blender_name]
        obj.rotation_mode = "QUATERNION"
        baseline_q = obj.rotation_quaternion.copy()

        # spin through the angles
        for angle in range(ang_min, ang_max + 1, ang_step):
            rotate_object(blender_name, axis, angle)
            render_and_log( os.path.join( results_path, f"obj_rotate_{obj_key}_{angle:+03d}.png") )

            # restore object rotation
            obj.rotation_quaternion = baseline_q.copy()


    # Object Sliding
    slide_min, slide_max, slide_step = scene_cfg["objects"]["slide"]
    for obj_key in chosen_objs:
        meta = solve_state["objs"][obj_key]
        blender_name = meta["obj"]

        # read the DOF matrix
        M = meta.get("dof_matrix_translation")
        if not M:
            print(f"{obj_key} has no translation DOF, skipping")
            continue
        
        # record original location
        obj = bpy.data.objects[blender_name]
        baseline_loc = obj.location.copy()

        for i in range(len(M)):
            # extract column vector
            axis = (M[0][i], M[1][i], M[2][i])

            # skip zero axis
            if abs(axis[0]) < 1e-6 and abs(axis[1]) < 1e-6 and abs(axis[2]) < 1e-6:
                continue
            
            for dist in range(int(slide_min*10), int(slide_max*10) + 1, int(slide_step*10)):
                dist /= 10
                slide_object(blender_name, axis, dist)
                render_and_log( os.path.join(results_path, f"obj_slide_{obj_key}_dof{i}_{dist:+.2f}.png") )
                obj.location = baseline_loc.copy()

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