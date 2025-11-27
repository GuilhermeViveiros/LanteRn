import torch
import os
from transformers import AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from dataclasses import dataclass
from typing import Optional

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

import logging
logger = logging.getLogger(__name__)

@dataclass
class LanternGenerateOutput:
    input_ids: torch.LongTensor
    latent_pred_values: torch.FloatTensor

def generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    gt_latent_embeds: Optional[torch.FloatTensor] = None,
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

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
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
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    input_embeds = model.get_input_embeddings()(input_ids)

    # -> Latent mode init
    in_latent_mode = False
    latent_start = False
    latent_end = False
    latent_num = 0
    latent_end_num = 0
    MAX_LATENT_LEN = model.config.latent_size
    latent_pred_values = []


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


        if in_latent_mode:
            latent_start = False
            # check if the next token is the latent end token
            if latent_num == MAX_LATENT_LEN:
                logger.info(f"Latent mode ended")
                latent_end = True
                latent_num = 0
                in_latent_mode = False
            else:
                logger.info(f"\033[90mlatent_num: {latent_num}\033[0m")
                latent_num += 1

        if latent_start_idx == next_tokens[0]:
            logger.info(f"Latent mode started")
            in_latent_mode = True
            latent_start = True        

        # get next tokens
        if latent_start:
            next_tokens = torch.tensor([latent_start_idx], device=next_tokens.device)
        elif in_latent_mode and not latent_start and not latent_end:
            next_tokens = torch.tensor([latent_pad_idx], device=next_tokens.device)
        elif latent_end:
            next_tokens = torch.tensor([latent_end_idx], device=next_tokens.device)
            latent_end = False
        else:
            next_tokens = next_tokens
        

        logger.info(f"next_tokens: {next_tokens}")
        for i, token_str in enumerate(next_tokens):
            # Print the token in color using ANSI escape codes (e.g., green)
            token_decoded = tokenizer.decode(token_str)
            logger.info(f"next token: at index {i} -> \033[92m{token_decoded}\033[0m <-")
        
    
        # get embedding of the next token
        if in_latent_mode and not latent_start and not latent_end:
        #if in_latent_mode and not latent_start and not latent_end:
            logger.info(f"using latent hidden states")        
            nb_latent_tokens = len(latent_pred_values)
            # next_token_embed = outputs.hidden_states[..., -1, :].unsqueeze(0)
            # TODO: Debugging purposes, we the ground truth latent embeddings to the generate function
            next_token_embed = gt_latent_embeds[...,(nb_latent_tokens),:].unsqueeze(0)
            latent_pred_values.append(next_token_embed)
            
            
        else:
            next_token_embed = model.get_input_embeddings()(next_tokens[:, None])
        
        input_embeds = torch.cat((input_embeds, next_token_embed), dim=1)
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

    if len(latent_pred_values) > 0:
        latent_pred_values = torch.cat(latent_pred_values, dim=1)
    else:
        latent_pred_values = None

    return LanternGenerateOutput(
        input_ids=input_ids,
        latent_pred_values=latent_pred_values,
    )