import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger("LantErn-Generate")

@dataclass
class LantErnGenerateOutput(GenerateDecoderOnlyOutput):
    # By inheriting from GenerateDecoderOnlyOutput, all its fields are inherited.
    # You only need to add new fields that are unique to this subclass.
    latent_embeds: Optional[torch.BFloat16Tensor] = None
    latent_mask: Optional[torch.BoolTensor] = None

def generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    gt_latent_embeds: Optional[torch.BFloat16Tensor] = None,
    perturbation: Optional[str] = None,
    **kwargs,
):

    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # check if use_cache is True
    use_cache = kwargs.get("use_cache", False)
    if not use_cache:
        logger.warning("\033[93mNot using cache. This may slow down generation a LOT!!!.\033[0m")

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, kwargs)

    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                print.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    input_embeds = model.get_input_embeddings()(input_ids)

    # -> Latent mode init
    in_latent_mode = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    latent_num = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)
    MAX_LATENT_LEN = model.config.latent_size
    latent_embeds = [[] for _ in range(batch_size)]
    latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    latent_start_idx = model.config.lvr_start_id
    latent_end_idx = model.config.lvr_end_id
    latent_pad_idx = model.config.lvr_sep_id

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"inputs_embeds": input_embeds})

        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # ---------------------------------------------------------
        # 1) Determine if the model entered latent mode
        # ---------------------------------------------------------
        latent_start_mask = ~in_latent_mode & (next_tokens == latent_start_idx)
        in_latent_mode[latent_start_mask] = True

        # ---------------------------------------------------------
        # 2) For samples already in_latent_mode:
        #    - Determine whether they still need pad tokens or should end
        # ---------------------------------------------------------
        need_pad_mask = in_latent_mode & (latent_num <= MAX_LATENT_LEN) & ~latent_start_mask
        latent_num[need_pad_mask] += 1

        # ---------------------------------------------------------
        # 3) Update latent counters if the latent mode terminated
        # ---------------------------------------------------------
        latent_end_mask = in_latent_mode & (latent_num > MAX_LATENT_LEN)
        if latent_end_mask.any():
            latent_num[latent_end_mask] = 0
            in_latent_mode[latent_end_mask] = False
            need_pad_mask[latent_end_mask] = False

        latent_mask = torch.cat((latent_mask, need_pad_mask.unsqueeze(1)), dim=1)

        # ---------------------------------------------------------
        # 4) Build the next_tokens
        #    priority: latent_sep -> latent_end -> next_tokens
        # ---------------------------------------------------------
        next_tokens = torch.where(need_pad_mask,  torch.full_like(next_tokens, latent_pad_idx),  next_tokens)
        next_tokens = torch.where(latent_end_mask, torch.full_like(next_tokens, latent_end_idx), next_tokens)

        # ---------------------------------------------------------
        # 5) Determine embedding for next token
        #    Latent positions get continuous hidden-state representations
        # ---------------------------------------------------------
        next_token_embed = model.get_input_embeddings()(next_tokens[:, None])
        if need_pad_mask.any():
            batch_indices = torch.nonzero(need_pad_mask, as_tuple=False).squeeze(1)
            if gt_latent_embeds is not None:
                latent_positions = latent_num[need_pad_mask] - 1
                next_latent_embed = gt_latent_embeds[batch_indices, latent_positions, :].unsqueeze(1)
            else:
                next_latent_embed = outputs.hidden_states[batch_indices, -1, :].unsqueeze(1)

            if perturbation == "zeros":
                next_latent_embed = torch.zeros_like(next_latent_embed)
            elif perturbation == "random":
                next_latent_embed = torch.randn_like(next_latent_embed)

            next_token_embed[batch_indices] = next_latent_embed.to(dtype=torch.bfloat16)
            for idx, embed in zip(batch_indices, next_latent_embed):
                latent_embeds[idx].append(embed.squeeze(0))

        # ---------------------------------------------------------
        # 6) Replace the input embeddings with the new embeddings
        # ---------------------------------------------------------
        input_embeds = next_token_embed if use_cache else torch.cat((input_embeds, next_token_embed), dim=1)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    if streamer is not None:
        streamer.end()

    if not latent_mask.any():
        latent_embeds = None

    if return_dict_in_generate:
        return LantErnGenerateOutput(
            sequences=input_ids,
            latent_embeds=latent_embeds,
            latent_mask=latent_mask,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
        return input_ids


def generate_skip_latent(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
    """Generation loop that suppresses latent reasoning blocks.

    When <|lvr_start|> is predicted, the next token is forced to <|lvr_end|>
    so the block is emitted as an empty <|lvr_start|><|lvr_end|> pair without
    entering latent mode or substituting any hidden-state embeddings.
    """

    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    use_cache = kwargs.get("use_cache", False)
    if not use_cache:
        logger.warning("\033[93mNot using cache. This may slow down generation a LOT!!!.\033[0m")

    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, kwargs)

    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                print.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    input_embeds = model.get_input_embeddings()(input_ids)

    latent_start_idx = model.config.lvr_start_id
    latent_end_idx = model.config.lvr_end_id

    # pending_end[i] = True means next token for sample i must be <|lvr_end|>
    pending_end = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"inputs_embeds": input_embeds})

        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # ---------------------------------------------------------
        # Force <|lvr_end|> one step after <|lvr_start|>, then arm
        # pending_end for any new <|lvr_start|> seen this step.
        # ---------------------------------------------------------
        next_tokens = torch.where(pending_end, torch.full_like(next_tokens, latent_end_idx), next_tokens)
        pending_end[:] = False
        pending_end[next_tokens == latent_start_idx] = True

        next_token_embed = model.get_input_embeddings()(next_tokens[:, None])
        input_embeds = next_token_embed if use_cache else torch.cat((input_embeds, next_token_embed), dim=1)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return LantErnGenerateOutput(
            sequences=input_ids,
            latent_embeds=None,
            latent_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
        return input_ids
