import bpy
import os
import json
import random
import math

# -----------------------------
# Config
# -----------------------------
output_dir = "./mental_rotation_dataset"  # absolute path for HPC
os.makedirs(output_dir, exist_ok=True)

num_samples = 10  # generate 10 samples for demo
objects = ["Cube", "Cube"]  # can add more meshes later
distractor_rotation_range = [(0, 360), (0, 360), (0, 360)]  # for X,Y,Z

# -----------------------------
# Helper functions
# -----------------------------
def setup_camera():
    """Ensure there is an active camera in the scene."""
    cams = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cams:
        bpy.context.scene.camera = cams[0]
    else:
        bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.1, 0, 0))
        bpy.context.scene.camera = bpy.context.object

def clear_scene():
    """Remove all objects except the camera."""
    bpy.ops.object.select_all(action='SELECT')
    for obj in bpy.context.selected_objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

def add_object(obj_type="Cube"):
    if obj_type == "Cube":
        bpy.ops.mesh.primitive_cube_add(size=2)
    elif obj_type == "Sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
    # add more shapes if needed
    return bpy.context.active_object

def set_rotation(obj, rot_xyz):
    # Blender uses radians
    obj.rotation_euler = [math.radians(rot) for rot in rot_xyz]

def render_image(filepath):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# -----------------------------
# Main generation loop
# -----------------------------
setup_camera()
dataset = []

for i in range(num_samples):
    clear_scene()
    
    # 1. Create reference object
    ref_obj = add_object("Cube")
    ref_rotation = [random.choice([0,90,180,270]) for _ in range(3)]
    set_rotation(ref_obj, ref_rotation)
    
    # 2. Render reference image
    ref_img_path = os.path.join(output_dir, f"sample_{i}_ref.png")
    render_image(ref_img_path)
    
    # 3. Create target object (rotated version)
    target_obj = add_object("Cube")
    # Random rotation to create positive (same) or negative (different) sample
    is_same = random.random() < 0.5
    if is_same:
        # rotate by random angle along Y axis
        rot_offset = random.choice([0,90,180,270])
        target_rotation = [(r + rot_offset)%360 for r in ref_rotation]
    else:
        target_rotation = [random.randint(0, 360) for _ in range(3)]
    
    set_rotation(target_obj, target_rotation)
    
    # Render target image
    target_img_path = os.path.join(output_dir, f"sample_{i}_target.png")
    render_image(target_img_path)
    
    # 4. Generate distractor images
    distractors = []
    for d in range(2):
        clear_scene()
        d_obj = add_object("Cube")
        d_rot = [random.randint(0,360) for _ in range(3)]
        set_rotation(d_obj, d_rot)
        d_img_path = os.path.join(output_dir, f"sample_{i}_distractor_{d}.png")
        render_image(d_img_path)
        distractors.append(d_img_path)
    
    # 5. Save metadata
    dataset.append({
        "sample_id": i,
        "category": "mental_rotation",
        "ref_image": ref_img_path,
        "target_image": target_img_path,
        "ref_rotation": ref_rotation,
        "target_rotation": target_rotation,
        "label": "Yes" if is_same else "No",
        "distractors": distractors,
        "latent_steps": [
            "infer 3D structure",
            "simulate rotation",
            "compare alignment"
        ]
    })

# Save dataset as JSON
with open(os.path.join(output_dir, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset generated with {num_samples} samples at {output_dir}")