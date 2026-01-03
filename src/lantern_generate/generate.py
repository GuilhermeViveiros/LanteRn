import torch
import os
from transformers import AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from dataclasses import dataclass
from typing import Optional
import logging
from itertools import chain

logger = logging.getLogger("LantErn-Generate")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer.add_tokens("<|lvr_start|>", special_tokens=False)
tokenizer.add_tokens("<|lvr_sep|>", special_tokens=False)
tokenizer.add_tokens("<|lvr_end|>", special_tokens=False)

@dataclass
class LantErnGenerateOutput:
    input_ids: torch.LongTensor
    latent_embeds: Optional[torch.BFloat16Tensor]
    latent_mask: Optional[torch.BoolTensor]

def generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    gt_latent_embeds: Optional[torch.BFloat16Tensor] = None,
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
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if model.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
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
    # latent_values: TODO: for now it only supports a fixed number of latent tokens per sample (MAX_LATENT_LEN)
    latent_embeds = [[] for _ in range(batch_size)] # every sample has its own latent embeds list (different sizes)
    latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)


    latent_start_idx = model.config.lvr_start_id
    latent_end_idx = model.config.lvr_end_id
    latent_pad_idx = model.config.lvr_sep_id

    tmp_value = 0

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"inputs_embeds": input_embeds})

        if is_prefill:
            # initial forward pass of the model (before the generation)
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            # generation-specific forward function
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue
   
        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
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

        # finished sentences should have their next token be a padding token
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
        #    - For samples that have terminated latent mode, set:
        #      - latent_num to 0
        #      - in_latent_mode to False
        #      - need_pad_mask to False
        #    - Store the latent mask
        # ---------------------------------------------------------
        latent_end_mask = in_latent_mode & (latent_num > MAX_LATENT_LEN)
        if latent_end_mask.any():
            latent_num[latent_end_mask] = 0
            in_latent_mode[latent_end_mask] = False
            need_pad_mask[latent_end_mask] = False

        latent_mask = torch.cat((latent_mask, need_pad_mask.unsqueeze(1)), dim=1)

        # if need_pad_mask count
        tmp_value += need_pad_mask.sum()
        

        # ---------------------------------------------------------
        # 4) Build the next_tokens
        #    priority: latent_sep -> latent_end -> next_tokens
        #    latent_start is handled in the previous step
        # ---------------------------------------------------------
        next_tokens = torch.where(need_pad_mask,  torch.full_like(next_tokens, latent_pad_idx),   next_tokens)
        next_tokens = torch.where(latent_end_mask,  torch.full_like(next_tokens, latent_end_idx), next_tokens)
        

        # ---------------------------------------------------------
        # 5) Determine embedding for next token
        #    We'll construct next_token_embed of shape (batch_size, 1, embed_dim)
        # ---------------------------------------------------------
        next_token_embed = model.get_input_embeddings()(next_tokens[:, None])
        # latent tokens get continuous representations
        if need_pad_mask.any():
            batch_indices = torch.nonzero(need_pad_mask, as_tuple=False).squeeze(1)
            if gt_latent_embeds is not None: # for debugging purposes (we can use the gt during the generation)
                latent_positions = latent_num[need_pad_mask] - 1
                next_latent_embed = gt_latent_embeds[batch_indices, latent_positions,:].unsqueeze(1) # (batch_size, 1, embed_dim)
            else:
                next_latent_embed = outputs.hidden_states[batch_indices, -1, :].unsqueeze(1) # (batch_size, 1, embed_dim)
            
            next_token_embed[batch_indices] = next_latent_embed.to(dtype=torch.bfloat16)
            # store the latent values (dynamic size per sample)
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

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    # if latent_mask is all False, don't return the latent values and mask (text-only prediction)
    if not latent_mask.any():
        latent_embeds = None
    #else:
        #latent_embeds = [embed for embed in latent_embeds if len(embed) > 0] # ignore empty lists
        #latent_embeds = list(chain.from_iterable(latent_embeds)) # nested list to single list
        #latent_embeds = torch.stack(latent_embeds, dim=0).to(dtype=torch.bfloat16)

    return LantErnGenerateOutput(input_ids=input_ids, latent_embeds=latent_embeds, latent_mask=latent_mask)