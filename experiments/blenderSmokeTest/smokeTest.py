# DO blender --background --python /path/to/test_blender.py

# test_blender.py
import bpy

# 1) Print Blender version
print("Blender version:", bpy.app.version_string)

# 2) Print the current scene name
scene = bpy.context.scene
print("Active scene:", scene.name)

# 3) Add a test cube to the scene
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
cube = bpy.context.object
print("Created object:", cube.name, "at", cube.location)

# 4) Clean up: remove the cube
bpy.ops.object.delete()

print("✅ Blender Python test succeeded!")