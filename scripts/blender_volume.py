"""Blender Cycles volume render of the vonkarman |omega| VDB sequence.

Glowing-volume look: density drives a CVD-safe emission ramp; Cycles on GPU.

Run (headless):
    blender -b -P scripts/blender_volume.py -- vdb_out renders
    # -> renders/frame_0001.png ...  then:
    # ffmpeg -framerate 30 -i renders/frame_%04d.png -pix_fmt yuv420p reconnection.mp4
"""
import bpy
import sys
import glob
import os

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
vdb_dir = argv[0] if argv else "vdb_out"
out_dir = argv[1] if len(argv) > 1 else "renders"
os.makedirs(out_dir, exist_ok=True)

vdbs = sorted(glob.glob(os.path.join(vdb_dir, "frame_*.vdb")))
if not vdbs:
    sys.exit(f"no frame_*.vdb in {vdb_dir}")
first = os.path.abspath(vdbs[0])

# Clean default scene.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import the VDB sequence as one Volume object (numbered files auto-detected).
bpy.ops.object.volume_import(filepath=first, use_sequence_detection=True)
vol = bpy.context.view_layer.objects.active
vol.data.frame_duration = len(vdbs)

# Emission volume shader: density -> ColorRamp (deep blue to hot white).
mat = bpy.data.materials.new("Omega")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
pv = nt.nodes.new("ShaderNodeVolumePrincipled")
attr = nt.nodes.new("ShaderNodeAttribute")
attr.attribute_name = "density"
ramp = nt.nodes.new("ShaderNodeValToRGB")
e = ramp.color_ramp.elements
e[0].position, e[0].color = 0.05, (0.0, 0.02, 0.12, 1.0)
e[1].position, e[1].color = 1.0, (1.0, 0.85, 0.5, 1.0)
mid = ramp.color_ramp.elements.new(0.45)
mid.color = (0.1, 0.45, 0.9, 1.0)
pv.inputs["Density"].default_value = 6.0
pv.inputs["Emission Strength"].default_value = 8.0
nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], pv.inputs["Emission Color"])
nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
vol.data.materials.clear()
vol.data.materials.append(mat)

# Camera looking slightly down the tubes.
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (3.0, -3.2, 2.2)
cam.rotation_euler = (1.05, 0.0, 0.78)
bpy.context.scene.camera = cam

# Dark world so the volume self-illuminates.
world = bpy.data.worlds.new("Dark")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
bpy.context.scene.world = world

# Cycles, GPU if available.
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception as exc:  # noqa: BLE001
    print(f"GPU setup skipped, using CPU: {exc}")
scene.cycles.samples = 96
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
scene.render.image_settings.file_format = "PNG"
scene.frame_start = 1
scene.frame_end = len(vdbs)
scene.render.filepath = os.path.join(out_dir, "frame_")
bpy.ops.render.render(animation=True)
