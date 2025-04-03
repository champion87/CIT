import os

import numpy as np
import torch
from cit_utils import image_to_tensor, show_latents, tensor_to_image, show_views
from project import UVProjection as UVP
from PIL import Image
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode


CAMERA_AZIMS=[-180, -135, -90, -45, 0, 45, 90, 135]
CAMERA_CENTERS=None
TOP_CAMERAS=True

def default_cameras():
	camera_poses = []
	attention_mask=[]

	cam_count = len(CAMERA_AZIMS)
	front_view_diff = 360
	back_view_diff = 360
	front_view_idx = 0
	back_view_idx = 0
	for i, azim in enumerate(CAMERA_AZIMS):
		if azim < 0:
			azim += 360
		camera_poses.append((0, azim))
		attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
		if abs(azim) < front_view_diff:
			front_view_idx = i
			front_view_diff = abs(azim)
		if abs(azim - 180) < back_view_diff:
			back_view_idx = i
			back_view_diff = abs(azim - 180)

	# Add two additional cameras for painting the top surfaces
	if TOP_CAMERAS:
		camera_poses.append((30, 0))
		camera_poses.append((30, 180))

		attention_mask.append([front_view_idx, cam_count])
		attention_mask.append([back_view_idx, cam_count+1])
  
	return camera_poses, attention_mask

# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="normal"):
	verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=render_size)
	masks = normals[...,3][:,None,...]
	masks = Resize((output_size//8,)*2, antialias=True)(masks)
	normals_transforms = Compose([
		Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True), 
		GaussianBlur(blur_filter, blur_filter//3+1)]
	)

	if cond_type == "normal":
		view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
		conditional_images = normals_transforms(view_normals)
	# Some problem here, depth controlnet don't work when depth is normalized
	# But it do generate using the unnormalized form as below
	elif cond_type == "depth":
		view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
		conditional_images = normals_transforms(view_depths)
	
	return conditional_images, masks

def get_extension(path):
	return os.path.splitext(path)[1].lower()

def set_cameras(uvp_app, camera_centers, camera_azims=[-180, -135, -90, -45, 0, 45, 90, 135], top_cameras=True):
	camera_poses = []
	# attention_mask=[]
	centers = camera_centers

	cam_count = len(camera_azims)
	front_view_diff = 360
	back_view_diff = 360
	front_view_idx = 0
	back_view_idx = 0
	for i, azim in enumerate(camera_azims):
		if azim < 0:
			azim += 360
		camera_poses.append((0, azim))
		# attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
		if abs(azim) < front_view_diff:
			front_view_idx = i
			front_view_diff = abs(azim)
		if abs(azim - 180) < back_view_diff:
			back_view_idx = i
			back_view_diff = abs(azim - 180)

	# Add two additional cameras for painting the top surfaces
	if top_cameras:
		camera_poses.append((30, 0))
		camera_poses.append((30, 180))

		# attention_mask.append([front_view_idx, cam_count])
		# attention_mask.append([back_view_idx, cam_count+1])
	# return camera_poses
	uvp_app.set_cameras_and_render_settings(camera_poses, centers=camera_centers, camera_distance=4.0)

# def prepare_uvp(tex_app_path, mesh_path_app, texture_rgb_size_app = 1024, device = "cuda:0"):
# 	uvp_app = UVP(texture_size=texture_rgb_size_app, render_size=512, sampling_mode="nearest", channels=3, device=device)
# 	# uvp_app.load_mesh(mesh_path_app, scale_factor=True, autouv=True)
# 	extension = get_extension(mesh_path_app)
# 	if extension == ".obj":
# 		uvp_app.load_mesh(mesh_path_app, scale_factor=True, autouv=False)
# 	elif extension == ".glb":
# 		uvp_app.load_glb_mesh(mesh_path_app, scale_factor=True, autouv=False)
# 	else:
# 		raise ValueError(f"Unsupported mesh file extension: {extension}")


# 	texture_image = Image.open(tex_app_path).convert('RGB')
# 	texture_tensor = image_to_tensor(texture_image)
# 	uvp_app.set_texture_map(texture_tensor)
 
# 	return uvp_app

def reshape_latents(latents):
    """
	I want the noise to be at index 0 and clear is at the last index
	I want to access the latents by [timestep, batch, channel, x, y]
    """
    # show_latents(latents.permute(1, 0, 2, 3, 4)[-1]) #This is the clear image
    return latents.permute(1, 0, 2, 3, 4)#[:-1]
 
def reshape_inverted_latents(latents):
	"""
	I want only the 100 noisy latents (all timesteps but 1)
	I want the noise to be at index 0 and clear is at the last index
	I want to access the latents by [timestep, batch, channel, x, y]
	"""
	return latents.permute(1, 0, 2, 3, 4)[1:].flip(dims=[-1])

def modify_bg(view:Image,mask:Image, bg:Image):
	"""
	Modify the background of the view using the mask and the background image
	"""
	res = bg.copy()
	res.paste(view, mask)
	return res

def mask_tensor_to_image(mask):
    mask = (mask * 255).clamp(0, 255).byte()
    mask_image = Image.fromarray(mask.cpu().numpy(), mode='L')
    return mask_image

def get_views_and_depth(tex_app_path, mesh_path_app, bg_path=None, camera_azims=[-180, -135, -90, -45, 0, 45, 90, 135], camera_centers=None, height=512,cond_type="depth"):
	"""
 	Returns a list of PIL Images, each is a view of the mesh with the texture applied, and a tensot, which is a collection of the depth maps of the mesh"""
	uvp_app = build_uvp(tex_app_path, mesh_path_app)
	set_cameras(uvp_app, camera_centers, camera_azims=camera_azims)
 
	app_views = uvp_app.render_textured_views() # List of 10 tensors, each is 1536x1536, but with 4 channels (last channel is mask)

	masks = [mask_tensor_to_image(view[-1]) for view in app_views]
	app_views = [tensor_to_image(view[:-1]) for view in app_views]
 
	extension = get_extension(mesh_path_app)
	conditional_images, _ = get_conditioning_images(uvp_app, height, cond_type=cond_type)

	if extension == ".glb":
		app_views = [view.transpose(Image.FLIP_TOP_BOTTOM) for view in app_views]
		conditional_images = torch.stack([image_to_tensor(tensor_to_image(view).transpose(Image.FLIP_TOP_BOTTOM)) for view in conditional_images])
		masks = [mask.transpose(Image.FLIP_TOP_BOTTOM) for mask in masks]
	
	if bg_path is not None:
		bg = Image.open(bg_path).convert('RGB')
		app_views = [modify_bg(view, mask, bg) for view, mask in zip(app_views, masks)]
  
	return app_views, conditional_images

def build_uvp(mesh_path, texture=None, texture_size=1024, render_size=512, sampling_mode="nearest", channels=3, device="cuda", camera_poses=None, camera_centers=None, mesh_transform={"scale":1}):
	"""The same as prepare_uvp but with more options"""
	uvp = UVP(texture_size=texture_size, render_size=render_size, sampling_mode=sampling_mode, channels=channels, device=device)
	extension = get_extension(mesh_path)
	if extension == ".obj":
		# raise ValueError(f"Unsupported mesh file extension: {extension}") 
		uvp.load_mesh(mesh_path, scale_factor=True, autouv=False)
	elif extension == ".glb":
		uvp.load_glb_mesh(mesh_path, scale_factor=True, autouv=False)
	else:
		raise ValueError(f"Unsupported mesh file extension: {extension}")
		assert False, "The mesh file format is not supported. Use .obj or .glb."

	if texture:
		texture_image = Image.open(texture) if type(texture) == str else texture
		texture_image = texture_image.convert("RGB")
		texture_tensor = (torch.from_numpy(np.array(texture_image)) / 255.0).permute(2, 0, 1)
		uvp.set_texture_map(texture_tensor)
 
	if camera_poses is None:
		camera_poses, _ = default_cameras()
	uvp.set_cameras_and_render_settings(camera_poses, centers=camera_centers, camera_distance=4.0)

	return uvp