"""Fix URDF mesh paths to absolute paths"""

import os
import re

# Paths
urdf_input = "/home/tatung/Desktop/Transform_bipedal/Transformer_IsaacLab/urdf/Transformer.urdf"
urdf_output = "/home/tatung/Desktop/Transform_bipedal/Transformer_IsaacLab/urdf/Transformer_fixed.urdf"
mesh_dir = "/home/tatung/Desktop/Transform_bipedal/Transformer_IsaacLab/meshes"

# Read URDF
with open(urdf_input, 'r') as f:
    content = f.read()

# Find all mesh filenames
mesh_pattern = r'filename="\.\.\/meshes\/([^"]+)"'
matches = re.findall(mesh_pattern, content)

print("🔍 Found meshes:")
for mesh in matches:
    print(f"  - {mesh}")

# Replace with absolute paths
def replace_mesh_path(match):
    mesh_file = match.group(1)
    absolute_path = f"{mesh_dir}/{mesh_file}"
    
    # Check if file exists
    if not os.path.exists(absolute_path):
        print(f"⚠️  WARNING: Mesh not found: {absolute_path}")
    else:
        print(f"✅ {mesh_file} → {absolute_path}")
    
    return f'filename="{absolute_path}"'

content_fixed = re.sub(mesh_pattern, replace_mesh_path, content)

# Save fixed URDF
with open(urdf_output, 'w') as f:
    f.write(content_fixed)

print(f"\n✅ Fixed URDF saved to: {urdf_output}")
print(f"\n📋 Summary:")
print(f"   - Total meshes: {len(matches)}")
print(f"   - Input:  {urdf_input}")
print(f"   - Output: {urdf_output}")