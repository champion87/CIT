from PIL import Image
import numpy as np
import math
import random
import torch
from torchvision.transforms import Resize, InterpolationMode


'''
	Encoding and decoding functions similar to diffusers library implementation
'''
@torch.no_grad()
def encode_latents(vae, imgs):
	imgs = (imgs-0.5)*2
	latents = vae.encode(imgs).latent_dist.sample()
	latents = vae.config.scaling_factor * latents
	return latents


@torch.no_grad()
def decode_latents(vae, latents):

	latents = 1 / vae.config.scaling_factor * latents

	image = vae.decode(latents, return_dict=False)[0]
	torch.cuda.current_stream().synchronize()

	image = (image / 2 + 0.5).clamp(0, 1)
	# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
	image = image.permute(0, 2, 3, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	
	return image


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
	# adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
	v1_4_latent_rgb_factors = torch.tensor([
		#   R        G        B
		[0.298, 0.207, 0.208],  # L1
		[0.187, 0.286, 0.173],  # L2
		[-0.158, 0.189, 0.264],  # L3
		[-0.184, -0.271, -0.473],  # L4
	], dtype=x.dtype, device=x.device)
	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents):
	result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
	resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
	result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
	textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
	result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
	return result_tex_rgb, result_tex_rgb_output


from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil
import torch
from PIL import Image

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/new_semester/cross-image-texturing/results/lidor"

def get_gpu_copy(tensor):
	"""Returns a copy of the tensor on the GPU.
	unused now, but might use it later."""
	return tensor.to("cuda:0", dtype=torch.float16)

def concat_images_vertically(images):
    # Get total height and maximum width
	total_height = sum(img.height for img in images)
	max_width = max(img.width for img in images)

	# Create a new blank image with the right size
	concatenated_img = Image.new("RGB", (max_width, total_height))

	# Paste images on top of each other
	y_offset = 0
	for img in images:
		concatenated_img.paste(img, (0, y_offset))
		y_offset += img.height

	return concatenated_img

def concat_images_horizontally(images):
    if type(images) == torch.Tensor:
        images = [tensor_to_image(img) for img in images]
    # Get total width and maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the right size
    concatenated_img = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    for img in images:
        concatenated_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return concatenated_img

def image_to_tensor(image):
    return (torch.from_numpy(np.array(image)) / 255.0).permute(2, 0, 1)

def tensor_to_image(tensor):
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def show_views(views, dest_dir=lidor_dir, save=True): # Deprecated, can't remember what it does
	result_images = []
	for view in views:
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		result_images.append(rgb_image)
	concatenated_image = np.concatenate(result_images, axis=1)
	res_image = numpy_to_pil(concatenated_image)[0]
	if save:
		res_image.save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
	return res_image
 
def save_all_views(views, dest_dir=lidor_dir):
	for i, view in enumerate(views):
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		numpy_to_pil(rgb_image)[0].save(f"{dest_dir}/face_view_{i}.jpg")
 
def show_mesh(uvp, dest_dir=lidor_dir, save=True, texture=None): #TODO what if the mesh is rendered in latent space?
	"""uvp can be a path to a saved model or a UVP object."""

	if type(uvp) == str:
		from uvp_utils import build_uvp
		# if not texture:
		# 	texture = Image.new("RGB", (1024, 1024), "white")
		uvp = build_uvp(uvp, texture)
	device = uvp.device
  
	uvp.to("cuda:0")
	views = uvp.render_textured_views()
	uvp.to(device)

	return show_views(views, dest_dir, save)
 
def show_latents(latents, dest_dir=lidor_dir, save=True, only_last=False):
	"""
	Latents can be a tensor of shape (N, L) or (L,), or path.
	"""
	if isinstance(latents, str):
		latents = torch.load(latents)
 
	if (len(latents.shape) == 3):
		latents = latents.unsqueeze(0)
  
	if only_last:
		latents = latents[-1]
  
	if (len(latents.shape) == 3):
		latents = latents.unsqueeze(0).unsqueeze(0)
	elif (len(latents.shape) == 4):
		latents = latents.unsqueeze(0)

	views = []
	for view in latents:
		view = view.to(torch.float16).to("cpu")
		decoded_latents = latent_preview(view)
		concatenated_image = np.concatenate(decoded_latents, axis=1)
		views.append(concatenated_image)
	concatenated_image = np.concatenate(views, axis=0)
	res_image = numpy_to_pil(concatenated_image)[0]
	if save:
		res_image.save(f"{dest_dir}/show_latent_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
	return res_image

def smvd_log(pipe, intermediate_results, i, t, num_timesteps, multiview_diffusion_end, 
			 view_fast_preview, tex_fast_preview, latent_tex, pred_original_sample):
	if view_fast_preview:
		decoded_results = []
		for latent_images in intermediate_results[-1]:
			images = latent_preview(latent_images.to(pipe._execution_device))
			images = np.concatenate([img for img in images], axis=1)
			decoded_results.append(images)
		result_image = np.concatenate(decoded_results, axis=0)
		numpy_to_pil(result_image)[0].save(f"{pipe.intermediate_dir}/step_{i:02d}.jpg")
	else:
		decoded_results = []
		for latent_images in intermediate_results[-1]:
			images = decode_latents(pipe.vae, latent_images.to(pipe._execution_device))
			images = np.concatenate([img for img in images], axis=1)
			decoded_results.append(images)
		result_image = np.concatenate(decoded_results, axis=0)
		numpy_to_pil(result_image)[0].save(f"{pipe.intermediate_dir}/step_{i:02d}.jpg")

	if not t < (1 - multiview_diffusion_end) * num_timesteps:
		if tex_fast_preview:
			tex = latent_tex.clone()
			texture_color = latent_preview(tex[None, ...])
			numpy_to_pil(texture_color)[0].save(f"{pipe.intermediate_dir}/texture_{i:02d}.jpg")
		else:
			pipe.uvp_rgb.to(pipe._execution_device)
			result_tex_rgb, result_tex_rgb_output = get_rgb_texture(pipe.vae, pipe.uvp_rgb, pred_original_sample)
			numpy_to_pil(result_tex_rgb_output)[0].save(f"{pipe.intermediate_dir}/texture_{i:02d}.png")
			pipe.uvp_rgb.to("cpu")
