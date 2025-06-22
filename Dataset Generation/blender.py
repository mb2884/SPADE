import bpy, os, glob, math, json, random
import numpy as np
from math import radians, tan, pi
from mathutils import Vector, Quaternion

# -----------------------------
# Configuration Parameters
# -----------------------------
config = {
    "input_dir": os.path.abspath("C:/Users/mbarr/OneDrive/Documents/Thesis/datasetgen/objs"),
    "output_dir": os.path.abspath("C:/Users/mbarr/OneDrive/Documents/Thesis/datasetgen/output_enhanced"),
    "hdri_dir": os.path.abspath("C:/Users/mbarr/OneDrive/Documents/Thesis/datasetgen/hdri"),
    "fill_fraction": 0.6,
    "min_fill_fraction": 0.3,
    "track_offset_fraction": 0.2,
    "frame_start": 1,
    "frame_end_options": [60],         # OPTIMIZED: Reduced from 250 to 60 frames (80% reduction)
    "resolution_x": 512,               # OPTIMIZED: Reduced from 1280 to 512
    "resolution_y": 512,               # OPTIMIZED: Reduced from 720 to 512 
    "render_engine": 'BLENDER_EEVEE_NEXT',  # OPTIMIZED: Standard EEVEE instead of EEVEE_NEXT
    "camera_radius_lambda": 2.0,
    "num_keys": 8,                     # OPTIMIZED: Reduced from 16 to 8 keyframes
    "specular_value": 1.0,
    "specular_roughness": 0.05,
    "matte_roughness": 1.0,
    "key_light_energy": 8000,
    "fill_light_energy": 4000,
    "hdri_strength": 0.7,
    "max_positioning_attempts": 3,     # OPTIMIZED: Reduced from 5 to 3 attempts
    "radius_reduction_factor": 0.15,
    "min_valid_frames_ratio": 0.7,
    "emergency_scale_factor": 1.5,
    "fov_adjustment_factors": [0.8, 0.6, 0.4],
    "fov_variation": (0.9, 1.1),       # ADDED: Missing parameter
    "max_objects": 9999,                # ADDED: Limit to first 100 objects for speed
    "optimize_render": True            # ADDED: Enable render optimizations
}

if not os.path.exists(config["output_dir"]):
    os.makedirs(config["output_dir"])

# -----------------------------
# GPU Rendering Setup + OPTIMIZATIONS
# -----------------------------
bpy.context.scene.render.engine = config["render_engine"]

# OPTIMIZED: Configure faster render settings
if config["optimize_render"]:
    # Set viewport and render samples lower
    if hasattr(bpy.context.scene.eevee, "taa_render_samples"):
        bpy.context.scene.eevee.taa_render_samples = 16  # Lower samples
    
    # Reduce shadow map resolution
    if hasattr(bpy.context.scene.eevee, "shadow_cube_size"):
        bpy.context.scene.eevee.shadow_cube_size = "512"
    
    # Disable features we don't need
    if hasattr(bpy.context.scene.eevee, "use_ssr"):
        bpy.context.scene.eevee.use_ssr = False  # Disable screen space reflections
        bpy.context.scene.eevee.use_ssr_refraction = False
    
    if hasattr(bpy.context.scene.eevee, "use_gtao"):
        bpy.context.scene.eevee.use_gtao = False  # Disable ambient occlusion
    
    # Optimize motion blur
    if hasattr(bpy.context.scene.eevee, "use_motion_blur"):
        bpy.context.scene.eevee.use_motion_blur = False

# Set up GPU if using Cycles
if config["render_engine"] in ['CYCLES']:
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 32  # OPTIMIZED: Reduced samples
    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get("cycles")
    if cycles_addon is not None:
        cycles_prefs = cycles_addon.preferences
        cycles_prefs.compute_device_type = 'OPTIX'  # or 'CUDA'
        for device in cycles_prefs.devices:
            device.use = True

# -----------------------------
# Helper Functions
# -----------------------------
def get_socket_by_name(node, target_name):
    """Case-insensitive lookup for a node input socket."""
    for socket in node.inputs:
        if socket.name.lower() == target_name.lower():
            return socket
    available = [socket.name for socket in node.inputs]
    raise Exception(f"Socket '{target_name}' not found. Available sockets: {available}")

def validate_object_size_in_view(obj, camera, min_fill_fraction=0.3):
    """Check if object fills minimum screen percentage"""
    corners_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    cam_inv = camera.matrix_world.inverted()
    corners_cam = [cam_inv @ corner for corner in corners_world]
    
    # Filter points in front of camera
    corners_cam = [p for p in corners_cam if p.z < 0]
    if not corners_cam:
        return False  # Object is behind camera
    
    # Project points to screen coordinates (-1 to 1 range)
    screen_coords = [(p.x/-p.z, p.y/-p.z) for p in corners_cam]
    xs = [p[0] for p in screen_coords]
    ys = [p[1] for p in screen_coords]
    
    if not xs or not ys:
        return False
    
    # Get dimensions in screen space
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Camera view dimensions
    aspect_ratio = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y
    view_width = 2 * tan(camera.data.angle / 2) * aspect_ratio
    view_height = 2 * tan(camera.data.angle / 2)
    
    # Calculate fill percentages
    width_fill = (max_x - min_x) / view_width
    height_fill = (max_y - min_y) / view_height
    
    # Object must fill minimum percentage in at least one dimension
    return max(width_fill, height_fill) >= min_fill_fraction

def scale_object_to_camera_fill(obj, camera, fill_fraction=0.8):
    """Scale object to fill camera view by given fraction."""
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select only the target object
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    corners_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    cam_inv = camera.matrix_world.inverted()
    corners_cam = [cam_inv @ corner for corner in corners_world]
    xs = [pt.x for pt in corners_cam]
    ys = [pt.y for pt in corners_cam]
    obj_width = max(xs) - min(xs)
    obj_height = max(ys) - min(ys)
    zs = [abs(pt.z) for pt in corners_cam]
    d = sum(zs) / len(zs)
    view_height = 2 * d * tan(camera.data.angle / 2)
    aspect_ratio = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y
    view_width = view_height * aspect_ratio
    
    desired_height = fill_fraction * view_height
    desired_width = fill_fraction * view_width
    
    scale_factor_v = desired_height / obj_height if obj_height != 0 else 1
    scale_factor_h = desired_width / obj_width if obj_width != 0 else 1
    scale_factor = min(scale_factor_v, scale_factor_h)
    
    obj.scale = (obj.scale.x * scale_factor,
                 obj.scale.y * scale_factor,
                 obj.scale.z * scale_factor)
    
    bpy.ops.object.transform_apply(scale=True)
    return scale_factor

def add_texture_to_material(mat, textures_folder):
    """Add diffuse texture to material if available."""
    for ext in ('*.jpg', '*.png', '*.exr'):
        files = glob.glob(os.path.join(textures_folder, ext))
        if files:
            texture_file = files[0]
            print("Using texture file:", texture_file)
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            tex_node = nodes.new('ShaderNodeTexImage')
            try:
                tex_node.image = bpy.data.images.load(texture_file)
            except Exception as e:
                print("Failed to load texture:", e)
                return mat
            bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
            if bsdf is not None:
                links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            break
    return mat

def setup_enhanced_lighting(hdri_path):
    """Set up improved three-point lighting with HDRI."""
    # Set up HDRI
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    bg = nodes.new('ShaderNodeBackground')
    output = nodes.new('ShaderNodeOutputWorld')
    
    try:
        env_tex.image = bpy.data.images.load(hdri_path)
    except Exception as e:
        print(f"Failed to load HDRI: {e}")
        return []
    
    # Set HDRI strength
    bg.inputs['Strength'].default_value = config["hdri_strength"]
    links.new(env_tex.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])
    
    # OPTIMIZED: Simpler lighting setup - just 2 lights instead of 3
    # Add key light (main light)
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
    key_light = bpy.context.object
    key_light.data.energy = config["key_light_energy"]
    key_light.data.size = 3.0
    key_light.name = "Key_Light"
    
    # Add fill light (softer, fills shadows)
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 1))
    fill_light = bpy.context.object
    fill_light.data.energy = config["fill_light_energy"]
    fill_light.data.size = 5.0
    fill_light.name = "Fill_Light"
    
    return [key_light, fill_light]

def center_object(obj):
    """Center object at world origin."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = Vector((0, 0, 0))

def compute_camera_radius(obj, lambda_factor):
    """Compute camera radius based on object size."""
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    return lambda_factor * max_dim

def set_bezier_interpolation(obj):
    """Set keyframe interpolation to BEZIER for smoother motion."""
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'BEZIER'
                # Adjust handles for smoother transitions
                keyframe.handle_left_type = 'AUTO_CLAMPED'
                keyframe.handle_right_type = 'AUTO_CLAMPED'

def animate_camera_random_walk(camera, target, frame_start, frame_end, num_keys, radius):
    """Create smoother camera path."""
    random_walk_info = {"key_frames": [], "spherical_coordinates": []}
    
    # Start with safer initial position
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0.3 * math.pi, 0.7 * math.pi)  # Avoid extreme angles
    
    # Generate keyframes with smoother transitions
    key_frames = np.linspace(frame_start, frame_end, num=num_keys, dtype=int)
    
    for i, f in enumerate(key_frames):
        # Calculate position on sphere
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        
        pos = Vector((x, y, z)) + target.location
        camera.location = pos
        
        # Point camera at target
        direction = (target.location - pos).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()
        
        # Insert keyframes
        camera.keyframe_insert(data_path="location", frame=f)
        camera.keyframe_insert(data_path="rotation_euler", frame=f)
        
        # Record keyframe info
        random_walk_info["key_frames"].append(int(f))
        random_walk_info["spherical_coordinates"].append({"theta": theta, "phi": phi})
        
        # Smaller, smoother changes between keyframes
        if i < num_keys - 1:
            max_delta = math.pi/18  # Smaller angle changes (10°)
            dtheta = random.uniform(-max_delta, max_delta)
            dphi = random.uniform(-max_delta, max_delta)
            
            # If previous movement was large, reduce this one
            if i > 0:
                prev_dtheta = random_walk_info["spherical_coordinates"][i]["theta"] - random_walk_info["spherical_coordinates"][i-1]["theta"]
                prev_dphi = random_walk_info["spherical_coordinates"][i]["phi"] - random_walk_info["spherical_coordinates"][i-1]["phi"]
                
                if abs(prev_dtheta) > math.pi/36:  # If previous change was >5°
                    dtheta *= 0.5  # Reduce this change
                
                if abs(prev_dphi) > math.pi/36:
                    dphi *= 0.5
            
            theta += dtheta
            phi += dphi
            
            # Keep phi in safer range
            phi = max(0.2 * math.pi, min(0.8 * math.pi, phi))
    
    # Set smooth Bezier interpolation
    set_bezier_interpolation(camera)
    
    return random_walk_info

def render_animation(output_filepath):
    """Render the current scene as an animation."""
    bpy.context.scene.render.filepath = output_filepath
    print("Rendering animation to:", output_filepath)
    bpy.ops.render.render(animation=True)
    print("Finished rendering to:", output_filepath)

# -----------------------------
# Main Processing Loop (Assets)
# -----------------------------
# Get top-level model folders (e.g., BarberShopChair_01)
model_folders = [f.path for f in os.scandir(config["input_dir"]) if f.is_dir()]
print(f"Found {len(model_folders)} model folders")

# OPTIMIZED: Limit the number of objects to process
if config["max_objects"] > 0 and len(model_folders) > config["max_objects"]:
    print(f"Limiting to {config['max_objects']} objects for faster processing")
    # Shuffle the list to get a random subset
    random.shuffle(model_folders)
    model_folders = model_folders[:config["max_objects"]]

metadata_log = []

for model_folder in model_folders:
    model_name = os.path.basename(model_folder)
    
    # Find the inner folder containing the blend file (e.g., BarberShopChair_01_2k)
    inner_folders = [f.path for f in os.scandir(model_folder) if f.is_dir()]
    
    if not inner_folders:
        print(f"No inner folders found in {model_folder}")
        continue
    
    blend_folder = inner_folders[0]  # Use the first inner folder
    
    # Find blend file in this folder
    blend_files = glob.glob(os.path.join(blend_folder, "*.blend"))
    if not blend_files:
        print(f"No blend file found in {blend_folder}")
        continue
    
    blend_file = blend_files[0]
    textures_folder = os.path.join(blend_folder, "textures")
    has_textures = os.path.isdir(textures_folder)
    
    # Process this model
    asset_meta = {
        "asset_name": model_name,
        "blend_file": os.path.basename(blend_file),
        "has_textures": has_textures
    }
    
    print(f"\n===== Processing asset: {model_name} =====")
    print(f"Using blend file: {blend_file}")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import model
    imported_objects = []
    with bpy.data.libraries.load(blend_file, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
    
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported_objects.append(obj)
    
    mesh_objs = [obj for obj in imported_objects if obj.type == 'MESH']
    if not mesh_objs:
        print("No mesh objects found in", blend_file)
        continue
    
    # Join meshes and center
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()
    target = bpy.context.active_object
    center_object(target)
    
    # Compute initial camera radius based on target size
    radius = compute_camera_radius(target, config["camera_radius_lambda"])
    asset_meta["camera_radius"] = radius
    
    # Create specular material
    specular_mat = bpy.data.materials.new(name="SpecularMaterial")
    specular_mat.use_nodes = True
    nodes = specular_mat.node_tree.nodes
    bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        raise Exception("No Principled BSDF node found in the material!")
    
    # Set material properties with enhanced specular settings
    spec_socket = get_socket_by_name(bsdf, "Specular IOR Level")
    spec_socket.default_value = config["specular_value"]  # Maximum specularity
    rough_socket = get_socket_by_name(bsdf, "Roughness")
    rough_socket.default_value = config["specular_roughness"]  # Very glossy
    
    # Add texture if available
    if has_textures:
        specular_mat = add_texture_to_material(specular_mat, textures_folder)
    
    # Apply material to object
    if target.data.materials:
        target.data.materials[0] = specular_mat
    else:
        target.data.materials.append(specular_mat)
    
    # Setup camera
    bpy.ops.object.camera_add(location=(radius, 0, 0))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    
    # Set up render parameters
    scene = bpy.context.scene
    scene.frame_start = config["frame_start"]
    scene.frame_end = random.choice(config["frame_end_options"])
    asset_meta["frame_end"] = scene.frame_end
    
    # Set up HDRI and enhanced lighting
    hdri_files = glob.glob(os.path.join(config["hdri_dir"], "*.hdr")) + glob.glob(os.path.join(config["hdri_dir"], "*.exr"))
    if hdri_files:
        chosen_hdri = random.choice(hdri_files)
        asset_meta["hdri_used"] = os.path.basename(chosen_hdri)
        
        # Set up enhanced three-point lighting
        lights = setup_enhanced_lighting(chosen_hdri)
    else:
        print("No HDRI files found, using default lighting.")
    
    # Initialize camera FOV with variation
    fov_mult = random.uniform(*config["fov_variation"])
    camera.data.angle *= fov_mult
    asset_meta["camera_fov_multiplier"] = fov_mult
    
    # Multiple camera positioning attempts
    max_attempts = config["max_positioning_attempts"]
    camera_path_valid = False
    final_path_info = None
    
    for attempt in range(max_attempts):
        print(f"Camera positioning attempt {attempt+1}/{max_attempts}")
        
        # Reduce camera radius for closer positioning in subsequent attempts
        adjusted_radius = radius * (1 - config["radius_reduction_factor"] * attempt)
        
        # Animate camera with given radius
        path_info = animate_camera_random_walk(
            camera, target, 
            config["frame_start"], scene.frame_end, 
            config["num_keys"], adjusted_radius
        )
        
        # Check multiple frames to ensure object is properly sized throughout animation
        valid_frames = 0
        check_frames = np.linspace(scene.frame_start, scene.frame_end, 10, dtype=int)
        
        for frame in check_frames:
            scene.frame_set(frame)
            if validate_object_size_in_view(target, camera, config["min_fill_fraction"]):
                valid_frames += 1
        
        # If most frames have good object visibility, accept this camera path
        valid_ratio = valid_frames / len(check_frames)
        print(f"Valid frames ratio: {valid_ratio:.2f} ({valid_frames}/{len(check_frames)})")
        
        if valid_ratio >= config["min_valid_frames_ratio"]:
            camera_path_valid = True
            final_path_info = path_info
            break
        
        # If last attempt, force scale object more aggressively
        if attempt == max_attempts - 1 and not camera_path_valid:
            print("Last attempt unsuccessful. Using emergency scaling.")
            scale_factor = config["emergency_scale_factor"]
            
            # Deselect all objects
            bpy.ops.object.select_all(action='DESELECT')
            
            # Select only target
            target.select_set(True)
            bpy.context.view_layer.objects.active = target
            
            # Apply aggressive scaling
            target.scale = (target.scale.x * scale_factor, 
                          target.scale.y * scale_factor,
                          target.scale.z * scale_factor)
            bpy.ops.object.transform_apply(scale=True)
            
            # Use the last camera path
            final_path_info = path_info
    
    # After positioning camera, try adjusting FOV if object still too small
    scene.frame_set(scene.frame_start)  # Set to first frame for FOV adjustment
    if not validate_object_size_in_view(target, camera, config["min_fill_fraction"]):
        original_fov = camera.data.angle
        
        for zoom_factor in config["fov_adjustment_factors"]:
            print(f"Trying FOV adjustment factor: {zoom_factor}")
            camera.data.angle = original_fov * zoom_factor
            
            # Check if adjustment improves visibility
            if validate_object_size_in_view(target, camera, config["min_fill_fraction"]):
                print(f"FOV adjustment successful with factor {zoom_factor}")
                break
    
    # Record camera path information
    asset_meta["camera_path"] = final_path_info
    asset_meta["camera_positioning_successful"] = camera_path_valid
    
    # Final scale adjustment to fill camera view
    scale_factor = scale_object_to_camera_fill(target, camera, fill_fraction=config["fill_fraction"])
    asset_meta["scale_factor"] = scale_factor
    
    # Configure render settings with higher resolution
    scene.render.resolution_x = config["resolution_x"]
    scene.render.resolution_y = config["resolution_y"]
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    
    # Render specular version
    specular_video = os.path.join(config["output_dir"], model_name + "_specular.mp4")
    render_animation(specular_video)
    
    # Change material to non-specular
    spec_socket.default_value = 0.0  # No specular
    rough_socket.default_value = config["matte_roughness"]  # Maximum roughness for diffuse look
    
    # Render non-specular version
    matte_video = os.path.join(config["output_dir"], model_name + "_no_specular.mp4")
    render_animation(matte_video)
    
    # Save metadata
    asset_meta["specular_video"] = os.path.basename(specular_video)
    asset_meta["matte_video"] = os.path.basename(matte_video)
    metadata_log.append(asset_meta)
    
    print(f"===== Completed asset: {model_name} =====\n")

# Save all metadata
master_meta_path = os.path.join(config["output_dir"], "metadata_all.json")
with open(master_meta_path, 'w') as f:
    json.dump(metadata_log, f, indent=2)

print("Processing complete.")
