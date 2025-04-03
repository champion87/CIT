import glob

import psutil
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, ControlNetModel

from uvp_utils import reshape_latents

# suppress partial model loading warning
logging.set_verbosity_error()

from datetime import datetime
import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import gc
import torchvision.transforms as T

device = 'cuda'


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None, lora_weights_path=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            self.use_depth = True
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        if lora_weights_path is not None:
            self.unet.load_attn_procs(lora_weights_path)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

        # self.controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose").to("cuda")
        # self.controlnet = ControlNetModel.from_pretrained("/data/orp/repos/LooseControl/LooseControl", use_safetensors=True).to("cuda") # TODO this is not our controlnet...
        self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to("cuda")

        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path, square):
        if square:
            size = (512, 512)
        else:
            size = 512
        image_pil = T.Resize(size)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    def load_control_image(self, control_image_path, size=512):
        control_image_pil = T.Resize(size)(Image.open(control_image_path).convert("RGB"))
        control_image = T.ToTensor()(control_image_pil).unsqueeze(0).to(device)
        return control_image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, control_image, save_path):
        timesteps = reversed(self.scheduler.timesteps)
        mid = []
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # for i, t in enumerate(tqdm(timesteps)):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent,
                    t,
                    encoder_hidden_states=cond_batch.clone(),
                    controlnet_cond=control_image,
                    return_dict=False,
                )
                eps = self.unet(
                    latent,
                    t,
                    encoder_hidden_states=cond_batch,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                mid.append(latent)
                
        mid = torch.cat(mid, dim=0).to("cpu") # CIT TODO we are not returning mid for now (we dont care about it)
        return latent  # CIT 

    @torch.no_grad()
    def ddim_sample(self, x, cond, control_image, save_path):
        timesteps = self.scheduler.timesteps
        mid = []
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # for i, t in enumerate(tqdm(timesteps)):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        x,
                        t,
                        encoder_hidden_states=cond_batch.clone(),
                        controlnet_cond=control_image,
                        return_dict=False,
                    )
                    eps = self.unet(
                        x,
                        t,
                        encoder_hidden_states=cond_batch,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    
                    mid.append(x)

        mid = torch.cat(mid, dim=0).to("cpu")
        return x, mid # CIT uncomment this line
        # return x # CIT comments this line

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, control_path, save_path, timesteps_to_save,
                        inversion_prompt='', square=True):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path, square)
        latent = self.encode_imgs(image)
        _, _, h, w = latent.shape
        control_image = self.load_control_image(control_path, (h * 8, w * 8))

        inverted_x = self.inversion_func(cond, latent, control_image, save_path)
        torch.save(inverted_x, save_path)
        latent_reconstruction, recon_proccess= self.ddim_sample(inverted_x, cond, control_image, save_path) # CIT uncomment this line
        # latent_reconstruction = self.ddim_sample(inverted_x, cond, control_image, save_path) # CIT comments this line

        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction, recon_proccess # CIT uncomment this line
        # return rgb_reconstruction # CIT comments this line


def run(opt : argparse.Namespace):
    """
    Run the cross-image texturing process with ControlNet.
    Args:
        opt (argparse.Namespace): The options for running the process. It should contain the following attributes:
            - sd_version (str): The version of Stable Diffusion to use. Options are '2.1', '2.0', '1.5', or 'depth'.
            - save_steps (int): The number of timesteps to save during the process.
            - data_path (str): The path to the directory containing the input images.
            - control_image_path (str): The path to the directory containing the control images.
            - save_dir (str): The directory where the output files will be saved.
            - lora_weights_path (str): The path to the LoRA weights file.
            - steps (int): The number of steps for the extraction process.
            - inversion_prompt (str): The prompt to use for inversion.
            - square_size (int): The size of the square for processing.
    Returns:
        Saves the reconstructed images to the specified directory, and returns the path to the saved file.
    """
    

    # timesteps to save
    if opt.sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif opt.sd_version == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif opt.sd_version == '1.5':
        model_key = "runwayml/stable-diffusion-v1-5"
    elif opt.sd_version == 'depth':
        model_key = "stabilityai/stable-diffusion-2-depth"
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(opt.save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.save_steps,
                                                           strength=1.0,
                                                           device=device)

    images_path = sorted(glob.glob(os.path.join(opt.data_path, '*')))
    control_images_path = sorted(glob.glob(os.path.join(opt.control_image_path, '*')))

    os.makedirs(opt.save_dir, exist_ok=True)
    # print(images_path)

    recons = []
    model = Preprocess(device, sd_version=opt.sd_version, hf_key=None, lora_weights_path=opt.lora_weights_path)
    for image_path, control_path in tqdm(zip(images_path, control_images_path)):
        image_name = os.path.basename(image_path)
        save_path = os.path.join(opt.save_dir, image_name.replace(".png", ".pt"))

        recon_image, recon_proccess = model.extract_latents(data_path=image_path, # CIT uncomment this line
        # recon_image = model.extract_latents(data_path=image_path, # CIT comment this line
                                             control_path=control_path,
                                             num_steps=opt.steps,
                                             save_path=save_path,
                                             timesteps_to_save=timesteps_to_save,
                                             inversion_prompt=opt.inversion_prompt,
                                            square=opt.square_size,
                                            )
        recons.append(recon_proccess)
        
        
        T.ToPILImage()(recon_image[0]).save(os.path.join(opt.save_dir, f"recon_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}.jpg"))
        del recon_image
        # del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Memory Usage: {(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)):.2f} MB")

    recons = torch.stack(recons)
    recons = reshape_latents(recons)
    result_file_name = os.path.join(opt.save_dir, 'recons.pt')
    torch.save(recons, result_file_name)
    return result_file_name

def parse_args():
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,required=True)
                        # default='/home/orp/data_home/datasets/in2n/person-small/images')
    parser.add_argument('--control_image_path', type=str,required=True)
                        # default='/home/orp/data_home/datasets/in2n/person-small/skeletons')
    parser.add_argument('--save_dir', type=str,required=True)
                        # default='/home/orp/data_home/datasets/in2n/person-small/latents_pnp_rec_lora')
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--inversion_prompt', type=str, default="a man in a blue shirt and khaki pants standing in front of a white wall")
    parser.add_argument('--square_size', type=bool, default=False)
    parser.add_argument('--lora_weights_path', type=str, default=None)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    run(opt)