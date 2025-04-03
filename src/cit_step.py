import torch

'''
	Customized Step Function
	step on texture
	texture
'''
@torch.no_grad()
def step_tex(
		scheduler,
		uvp,
		model_output: torch.FloatTensor,
		timestep: int,
		prev_t: int,
		sample: torch.FloatTensor,
		texture: None,
		generator=None,
		return_dict: bool = True,
		guidance_scale = 1,
		main_views = [],
		hires_original_views = True,
		exp=None,
		cos_weighted=True,
		eta=None,
		is_app=False
):
	step_results = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        generator=generator,
        return_dict=return_dict,
    )
	pred_prev_sample = step_results['prev_sample']
	pred_original_sample = step_results['pred_original_sample']

	# Step texture
	if not is_app:
		prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
		alpha_prod_t = scheduler.alphas_cumprod[timestep]
		alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
		beta_prod_t = 1 - alpha_prod_t
		variance = scheduler._get_variance(timestep, prev_timestep)
		std_dev_t = eta * variance ** (0.5)

		if texture is None:
			sample_views = [view for view in sample]
			sample_views, texture, _ = uvp.bake_texture(views=sample_views, main_views=main_views, exp=exp)
			sample_views = torch.stack(sample_views, axis=0)[:,:-1,...]

		original_views = [view for view in pred_original_sample]
		original_views, original_tex, visibility_weights = uvp.bake_texture(views=original_views, main_views=main_views, exp=exp)
		uvp.set_texture_map(original_tex)
		original_views = uvp.render_textured_views()
		original_views = torch.stack(original_views, axis=0)[:,:-1,...]

		pred_tex_epsilon = (texture - alpha_prod_t ** (0.5) * original_tex) / beta_prod_t ** (0.5)
		pred_tex_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_tex_epsilon
		prev_tex = alpha_prod_t_prev ** (0.5) * original_tex + pred_tex_direction

	if not is_app:
		uvp.set_texture_map(prev_tex)
		prev_views = uvp.render_textured_views()
	pred_prev_sample = torch.clone(sample)
	if not is_app:
		for i, view in enumerate(prev_views):
			pred_prev_sample[i] = view[:-1]
		masks = [view[-1:] for view in prev_views]

	if not is_app:
		return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample, "prev_tex": prev_tex}
	else:
		return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample}
