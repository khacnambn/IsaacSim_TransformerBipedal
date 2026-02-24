"""Comprehensive USD joint analysis"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdGeom, Gf
import math

usd_path = "/home/tatung/Desktop/Transform_bipedal/transformer_nam/asset/Transformer_new.usd"
stage = Usd.Stage.Open(usd_path)

print("=" * 80)
print("🔍 COMPREHENSIVE USD JOINT ANALYSIS")
print("=" * 80)

# Find all revolute joints
joints_data = []

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        
        # Get joint properties
        name = prim.GetName()
        path = prim.GetPath()
        
        # Get axis
        axis_attr = joint.GetAxisAttr()
        axis = axis_attr.Get() if axis_attr else "NONE"
        
        # Get limits
        lower_attr = joint.GetLowerLimitAttr()
        upper_attr = joint.GetUpperLimitAttr()
        lower = lower_attr.Get() if lower_attr else None
        upper = upper_attr.Get() if upper_attr else None
        
        # Get body0 and body1 (parent and child links)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        
        joints_data.append({
            'name': name,
            'path': str(path),
            'axis': axis,
            'lower': lower,
            'upper': upper,
            'body0': str(body0[0]) if body0 else "NONE",
            'body1': str(body1[0]) if body1 else "NONE",
        })

# Sort by name for comparison
joints_data.sort(key=lambda x: x['name'])

# Print detailed comparison
print("\n📋 JOINT DETAILS (sorted alphabetically):")
print("-" * 80)

for i, joint in enumerate(joints_data):
    print(f"\n[{i}] {joint['name']}:")
    print(f"    Path:   {joint['path']}")
    print(f"    Axis:   {joint['axis']}")
    
    if joint['lower'] is not None and joint['upper'] is not None:
        print(f"    Limits: [{math.degrees(joint['lower']):.1f}°, {math.degrees(joint['upper']):.1f}°]")
    else:
        print(f"    Limits: NONE")
    
    print(f"    Body0:  {joint['body0']}")
    print(f"    Body1:  {joint['body1']}")

# Compare LEFT vs RIGHT
print("\n" + "=" * 80)
print("🔄 LEFT vs RIGHT COMPARISON")
print("=" * 80)

pairs = [
    ("Bubleft", "Bubright"),
    ("Hipleft", "Hipright"),
    ("Kneeleft", "Kneeright"),
    ("Footleft", "Footright"),
]

for left_name, right_name in pairs:
    left = next((j for j in joints_data if j['name'] == left_name), None)
    right = next((j for j in joints_data if j['name'] == right_name), None)
    
    if left and right:
        print(f"\n🔹 {left_name} vs {right_name}:")
        
        # Compare axis
        if left['axis'] == right['axis']:
            print(f"   ✅ Axis SAME: {left['axis']}")
        else:
            print(f"   ⚠️  Axis DIFFERENT:")
            print(f"      Left:  {left['axis']}")
            print(f"      Right: {right['axis']}")
        
        # Compare limits
        if left['lower'] == right['lower'] and left['upper'] == right['upper']:
            print(f"   ✅ Limits SAME: [{math.degrees(left['lower']):.1f}°, {math.degrees(left['upper']):.1f}°]")
        else:
            print(f"   ⚠️  Limits DIFFERENT:")
            if left['lower'] is not None:
                print(f"      Left:  [{math.degrees(left['lower']):.1f}°, {math.degrees(left['upper']):.1f}°]")
            if right['lower'] is not None:
                print(f"      Right: [{math.degrees(right['lower']):.1f}°, {math.degrees(right['upper']):.1f}°]")

# Check base frame orientation
print("\n" + "=" * 80)
print("📐 BASE FRAME CHECK")
print("=" * 80)

base_prim = stage.GetPrimAtPath("/World/Transformer")
if base_prim.IsValid():
    xform = UsdGeom.Xformable(base_prim)
    ops = xform.GetOrderedXformOps()
    
    print(f"\nBase prim: {base_prim.GetPath()}")
    print(f"Transform ops:")
    for op in ops:
        print(f"   {op.GetOpType()}: {op.Get()}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

simulation_app.close()