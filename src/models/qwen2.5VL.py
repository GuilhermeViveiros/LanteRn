from transformers import CrossEntropyLoss
from transformers import Qwen2_5_VLCausalLMOutputWithPast
from typing import Optional, Union, Any, Tuple, List, Dict, Unpack
from transformers.modeling_utils import Cache
import torch




'''
    Coconut mode
    No additional Head
    Custom implementation of LantErn on Qwen2.5VL model
'''
def lantern_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lantent_values: Optional[torch.Tensor] = None, # Custom LantErn feature: latent values for the visual reasoning
    latent_grid_thw: Optional[torch.LongTensor] = None, # Custom LantErn feature: latent grid dim for the visual reasoning
    latent_hidden_states: Optional[torch.Tensor] = None, # Custom LantErn feature: latent hidden states for the visual reasoning
    **kwargs: Unpack[Any],
) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if lantent_values is not None:
        # Get batch_size early to avoid repeated shape access
        batch_size = inputs_embeds.shape[0]
        # Ensure dtype conversion happens early and only once
        latent_visual_embeds = lantent_values.to(dtype=self.visual.dtype, device=inputs_embeds.device)
        latent_image_embeds = self.visual(latent_visual_embeds, latent_grid_thw)
        
        n_latent_tokens = (input_ids == self.config.lvr_sep_id).sum().item()
        n_latent_features = latent_visual_embeds.shape[0]
        hidden_states = latent_image_embeds.shape[-1]
        
        # for now we assume a fixed size for latent reasoning tokens (4)
        # so the latent features (reallty the patch features) should be reduce to 4 patches
        if n_latent_tokens != n_latent_features:
            assert n_latent_tokens < n_latent_features
            # we will average the latent features to get the 4 patches
            latent_image_embeds = latent_image_embeds.view(batch_size, -1, hidden_states)
            # TODO: we need better ways to encode the latent tokens, for now just use 4
            # 1. ensure its divisible by 4
            leave_out_patches = latent_image_embeds.shape[1] % self.model.config.latent_size
            if leave_out_patches > 0:
                latent_image_embeds = latent_image_embeds[:, :-leave_out_patches, :]
            # 2. group the latent features into 4 patches - optimized using reshape + mean
            group_size = latent_image_embeds.shape[1] // self.model.config.latent_size
            # Reshape to group patches together, then compute mean along the group dimension
            # Shape: (batch_size, n_groups, group_size, hidden_states) -> (batch_size, n_groups, hidden_states)
            latent_image_embeds = latent_image_embeds.view(
                batch_size, self.model.config.latent_size, group_size, hidden_states
            ).mean(dim=2, keepdim=False).contiguous()
            latent_image_embeds = latent_image_embeds.view(-1, hidden_states)

        n_latent_features = latent_image_embeds.shape[0]

        if n_latent_tokens != n_latent_features:
            raise ValueError(
                f"LantErn latent features and visual tokens do not match: tokens: {n_latent_tokens}, features {n_latent_features}"
            )
        
        # Optimized mask creation: avoid intermediate tensors, use expand_as for efficiency
        # Create mask directly on the correct device to avoid device transfer
        mask = (input_ids == self.config.lvr_sep_id).to(inputs_embeds.device)
        # Use expand_as instead of expand with shape tuple - more efficient
        latent_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
        
        # Ensure dtype matches before masked_scatter
        latent_image_embeds = latent_image_embeds.to(dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(latent_mask, latent_image_embeds)
        
    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    
    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        # hidden_states=outputs.hidden_states,
        hidden_states=hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
        inputs_embeds=inputs_embeds,
    )