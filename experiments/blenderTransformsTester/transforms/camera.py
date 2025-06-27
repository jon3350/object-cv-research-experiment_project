import bpy
from math import radians

def zoom_camera(camera_name: str, delta_mm: float):
    cam = bpy.data.objects[camera_name].data
    cam.lens += delta_mm

def zoom_camera_get_baseline(camera_name: str):
    cam = bpy.data.objects[camera_name].data
    return cam.lens

def zoom_camera_set(camera_name: str, baseline_mm: float, delta_mm: float):
    cam = bpy.data.objects[camera_name].data
    cam.lens = baseline_mm + delta_mm

def rotate_camera(camera_name: str, yaw: float = 0, pitch: float = 0, roll: float = 0):
    obj = bpy.data.objects[camera_name]
    # in blender the camera looks down the z-axis
    obj.rotation_euler[2] += radians(roll) # Z
    obj.rotation_euler[1] += radians(yaw)  # Y
    obj.rotation_euler[0] += radians(pitch) # Z

def rotate_camera_get_baseline(camera_name: str):
    obj = bpy.data.objects[camera_name]
    return obj.rotation_euler.copy()

def rotate_camera_set(camera_name: str, baseline, yaw: float = 0, pitch: float = 0, roll: float = 0):
    obj = bpy.data.objects[camera_name]
    # in blender the camera looks down the z-axis
    obj.rotation_euler[2] = baseline[2] + radians(roll) # Z
    obj.rotation_euler[1] = baseline[1] + radians(yaw)  # Y
    obj.rotation_euler[0] = baseline[0] + radians(pitch) # Z
