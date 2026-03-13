import bpy
import os
from mathutils import Vector

# =========================
# Settings
# =========================
ply_path = "outputs/lego/iris/demo/tetrahedron_soup.ply"
object_name = "tetrahedron_soup"
lattice_name = "Lattice"
output_dir = "outputs/lego/iris/demo/camera_path"

scale_factor = 3.0
initial_translation = Vector([-1.5, -1.5, -1.5])
z_offset = 0.6

# =========================
# Ensure output directory exists
# =========================
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =========================
# Import PLY
# =========================
bpy.ops.wm.ply_import(filepath=ply_path)

obj = bpy.context.selected_objects[0]
obj.name = object_name

# =========================
# Apply simulation transform
# =========================
obj.scale = (scale_factor, scale_factor, scale_factor)
obj.location += initial_translation
obj.location.z += z_offset

# Select the object and set it as active (required for applying transforms)
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Apply the transformations
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

# =========================
# Bind to existing lattice
# =========================
lattice = bpy.data.objects[lattice_name]

mod = obj.modifiers.new(name="LatticeDeform", type='LATTICE')
mod.object = lattice

# =========================
# Frame range
# =========================
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# =========================
# Export frames
# =========================
for frame in range(start_frame, end_frame + 1):

    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

    # Save transform
    original_loc = obj.location.copy()
    original_scale = obj.scale.copy()

    # =========================
    # Export transform
    # =========================
    obj.location += Vector([0.5, 0.5, 0.5])
    obj.scale = original_scale / scale_factor

    filename = f"{(frame - 1):05d}.ply"
    filepath = os.path.join(output_dir, filename)

    bpy.ops.wm.ply_export(
        filepath=filepath,
        apply_modifiers=True,
        export_selected_objects=True,
        export_normals=True,
        export_uv=False,
        export_colors='NONE',
        export_attributes=False,
        export_triangulated_mesh=False,
        ascii_format=True,
        forward_axis='Y',
        up_axis='Z'
    )

    print(f"Exported frame {frame} to {filepath}")

    # =========================
    # Restore simulation transform
    # =========================
    obj.location = original_loc
    obj.scale = original_scale