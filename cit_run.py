from datetime import datetime
# Store script start time
script_start_time = datetime.now()

def log(msg: str):
    elapsed_time = datetime.now() - script_start_time
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    print(f"{datetime.now().strftime('%d%b%Y-%H%M%S')} (+{int(minutes)}m{int(seconds)}s) cit_run: {msg}")

log("Starting script")

import torch
log("imported torch")

from cit_configs import *
opt = parse_config()
log("parsed config")

import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from typing import List
from IPython.display import display
from datetime import datetime
from shutil import copy
from PIL import Image

log("imported other modules")

# This part is copied from SyncMVD/run_experiment.py and prepares the pipeline.
# Need to make sure that the pipe receives the two meshes and appearance texture instead of only one mesh.
# 'app' is a shortcut for 'appearance'

def resize_to_width(img, width):
    """Resizes an image to the given width while maintaining aspect ratio."""
    w, h = img.size
    new_height = int(h * (width / w))
    return img.resize((width, new_height))

def make_dirs_for_app_proccessing(output_dir):
	inversion_dir = join(output_dir, "inversion")
	cond_dir = join(output_dir, "cond")
	data_dir = join(output_dir, "data") # 'data' is the terminology of invertsion_with_controlnet for the views.

	if not isdir(inversion_dir):
		os.mkdir(inversion_dir)
	if not isdir(cond_dir):
		os.mkdir(cond_dir)
	if not isdir(data_dir):
		os.mkdir(data_dir)
  
	return inversion_dir, cond_dir, data_dir

def save_list_of_tensors(base_name:str, tensor_list, output_dir:str):
	for i, tensor in enumerate(tensor_list):
		torch.save(tensor, join(output_dir, f"{base_name}_{i}.pt"))

def save_list_of_images(base_name:str, image_list:List[Image.Image], output_dir:str):
	for i, image in enumerate(image_list):
		image.save(join(output_dir, f"{base_name}_{i}.png"))

def prepare_appearance(prompt, steps, tex_app, mesh_app, output_dir, seed, bg_path=None, invert_with_controlnet=True, cond_type="depth", camera_azims=None):
	"""Inverts the appearance of the object and saves the views and depth conditioning images."""
	from cit_utils import tensor_to_image, concat_images_horizontally
	from uvp_utils import get_views_and_depth
	from argparse import Namespace
	if invert_with_controlnet:
		from inversion_with_controlnet import run
	else:
		raise NotImplementedError("Inversion without controlnet is no longer supported.")
		# from inversion_save_latents_wo_cn import run
 
	inversion_dir, cond_dir, data_dir = make_dirs_for_app_proccessing(output_dir)
	cond_app_path = join(cond_dir, "cond_app.pt")

	print(f"Saving appearance views to {data_dir}")
	print(f"Saving appearance conditioning images to {cond_dir}")
	print(f"Saving inverted appearance latents to {inversion_dir}")
 
	app_views, app_conds = get_views_and_depth(tex_app_path=tex_app, mesh_path_app=mesh_app, camera_azims=camera_azims, cond_type=cond_type, bg_path=bg_path)
	save_list_of_images("view", app_views, data_dir)
	
	depth_conditional_images = [tensor_to_image(image) for image in app_conds]
	save_list_of_images("cond", depth_conditional_images, cond_dir)


	# Create opt object directly
	opt = Namespace(
		data_path=data_dir,
		control_image_path=cond_dir,
		save_dir=inversion_dir,
		sd_version='1.5',
		seed=seed,
		steps=steps,
		save_steps=1000,  # or another value if needed
		inversion_prompt=prompt,
		lora_weights_path=None,
		square_size=False,
		extract_reverse=False,
	)
	recon_latents_path = run(opt) # Should look like "inversion/latents.pt"
 
	concat_images_horizontally(app_views).save(join(data_dir, "all_views.png"))
	torch.save(app_conds, cond_app_path)
 
	return recon_latents_path, cond_app_path

def get_paths(opt):
	if opt.mesh_config_relative:
		mesh_path = join(dirname(opt.config), opt.mesh)
		mesh_path_app = join(dirname(opt.config), opt.mesh_app)
		tex_app = join(dirname(opt.config), opt.tex_app)
		latents_save_path = join(dirname(opt.config), opt.latents_save_path)
		cond_app_path = join(dirname(opt.config), opt.cond_app_path)
		bg_path = join(dirname(opt.config), opt.bg_path)
	else:
		mesh_path = abspath(opt.mesh)
		mesh_path_app = abspath(opt.mesh_app)
		tex_app = abspath(opt.tex_app)
		latents_save_path = abspath(opt.latents_save_path)
		cond_app_path = abspath(opt.cond_app_path)
		bg_path = abspath(opt.bg_path)
  
	if opt.output:
		output_root = abspath(opt.output)
	else:
		output_root = dirname(opt.config)
  
	return mesh_path, mesh_path_app, tex_app, latents_save_path, cond_app_path, bg_path, output_root

def make_output_dir(output_root, opt):
	output_name_components = []
	if opt.prefix and opt.prefix != "":
		output_name_components.append(opt.prefix)
	if opt.use_mesh_name:
		mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
		output_name_components.append(mesh_name)

	if opt.timeformat and opt.timeformat != "":
		output_name_components.append(datetime.now().strftime(opt.timeformat))
	output_name = "_".join(output_name_components)
	output_dir = join(output_root, output_name)

	if not isdir(output_dir):
		os.mkdir(output_dir)
	else:
		print(f"Results exist in the output directory, use time string to avoid name collision.")
		exit(0)
	
	return output_dir


mesh_path, mesh_path_app, tex_app, latents_save_path, cond_app_path, bg_path, output_root = get_paths(opt)
log("Got paths")

output_dir = make_output_dir(output_root, opt)	
log(f"Saving to {output_dir}")

if not opt.latents_load:
	log("Calculating new inverted latents and appearance conditioning images.")
	latents_save_path, cond_app_path = prepare_appearance(opt.prompt, opt.steps, tex_app, mesh_path_app, output_dir, opt.seed, bg_path, opt.invert_with_controlnet, opt.cond_type, opt.camera_azims)
	log("Done calculating new inverted latents and appearance conditioning images.")
else:
    log("Using provided latents and appearance conditioning images.")


if opt.preview:
	from cit_utils import show_latents, concat_images_vertically, concat_images_horizontally, show_mesh
	log("imported from cit_utils")
	app_depth = concat_images_horizontally(torch.load(cond_app_path))
	log("Loaded app_depth")
	app_inverted = resize_to_width(show_latents(latents_save_path, output_dir, save=False, only_last=True), app_depth.width)
	log("Loaded app_inverted")
	app_views = show_mesh(mesh_path_app, output_dir, save=False, texture=tex_app)
	log("Loaded app_views")
	target_depth = show_mesh(mesh_path, output_dir, save=False)
	log("Loaded target_depth")
	preview_img = concat_images_vertically([app_depth, app_inverted, app_views, target_depth])
	preview_img.save(join(output_dir, "preview.jpg"))
	log("Saved preview image.")
 
if opt.task == "preview_only" or opt.task == "invert":
	exit(0)

# Slow imports here
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers.training_utils import set_seed
from CIA.appearance_transfer_model import AppearanceTransferModel
if opt.task == "smvd":
    from SyncMVD.src.pipeline import StableSyncMVDPipeline
elif opt.task == "cit":
	from pipeline import StableSyncMVDPipeline

log("Imported diffusers and pipeline")


copy(opt.config, join(output_dir, "config.yaml"))

logging_config = {
	"output_dir":output_dir, 
	# "output_name":None, 
	# "intermediate":False, 
	"log_interval":opt.log_interval,
	"view_fast_preview": opt.view_fast_preview,
	"tex_fast_preview": opt.tex_fast_preview,
	}


# Set up the model and the pipeline

if opt.cond_type == "normal":
	# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16)
	raise NotImplementedError("Normal controlnet is not supported yet.")
elif opt.cond_type == "depth":
	controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)			
pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
syncmvd = StableSyncMVDPipeline(**pipe.components)

if True or opt.do_cit_injection: #TODO delete the 'if' later
	model_cfg = RunConfig(opt.prompt)
	set_seed(model_cfg.seed)
	# Modifying the attention proccesor of the pipe has a wierd semantic - it is done by initializing the model with the pipe.
	# I hope that even after the attention processor is modified, it can still be used by smvd as if nothing ever happened.
	model = AppearanceTransferModel(model_cfg, pipe=syncmvd)
	assert id(syncmvd) == id(model.pipe), f"syncmvd and model.pipe should be the same object." 

log("Model is set up")


# Run the SyncMVD pipeline
result_tex_rgb, textured_views, v = syncmvd(
	prompt=opt.prompt,
	height=opt.latent_view_size*8,
	width=opt.latent_view_size*8,
	num_inference_steps=opt.steps,
	guidance_scale=opt.guidance_scale,
	negative_prompt=opt.negative_prompt,
	
	generator=torch.manual_seed(opt.seed),
	max_batch_size=48,
	controlnet_guess_mode=opt.guess_mode,
	controlnet_conditioning_scale = opt.conditioning_scale,
	controlnet_conditioning_end_scale= opt.conditioning_scale_end,
	control_guidance_start= opt.control_guidance_start,
	control_guidance_end = opt.control_guidance_end,
	guidance_rescale = opt.guidance_rescale,
	use_directional_prompt=True,

	mesh_path=mesh_path,
	mesh_transform={"scale":opt.mesh_scale},
	mesh_autouv=not opt.keep_mesh_uv,

	camera_azims=opt.camera_azims,
	top_cameras=not opt.no_top_cameras,
	texture_size=opt.latent_tex_size,
	render_rgb_size=opt.rgb_view_size,
	texture_rgb_size=opt.rgb_tex_size,
	multiview_diffusion_end=opt.mvd_end,
	exp_start=opt.mvd_exp_start,
	exp_end=opt.mvd_exp_end,
	ref_attention_end=opt.ref_attention_end,
	shuffle_background_change=opt.shuffle_bg_change,
	shuffle_background_end=opt.shuffle_bg_end,

	logging_config=logging_config,
	cond_type=opt.cond_type,
 
	# CIT arguments
	tex_app_path=tex_app,	
	mesh_path_app=mesh_path_app,
	mesh_transform_app={"scale":opt.mesh_scale},
	mesh_autouv_app=not opt.keep_mesh_uv,
	latents_save_path=latents_save_path,
	cond_app_path=cond_app_path,
	perform_swap=opt.do_cit_injection,
	)

log("SyncMVD pipeline is done running")