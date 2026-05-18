import logging

import torch
from termcolor import colored

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")


def set_latent_tokens(processor, model, latent_size: int, special_tokens: bool = True):
    # add special tokens for LantErn
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=special_tokens)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=special_tokens)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=special_tokens)
    # set the latent size
    processor.latent_size = latent_size
    model.config.latent_size = latent_size

    model.config.additional_special_tokens = [
        "<|lvr_start|>",
        "<|lvr_sep|>",
        "<|lvr_end|>"
    ]

    # resize the model embeddings size
    model.resize_token_embeddings(len(processor.tokenizer))

    # get the ids of the special tokens -> used for the model and processor
    lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    processor.lvr_start_id, processor.lvr_sep_id, processor.lvr_end_id = lvr_start_id, lvr_sep_id, lvr_end_id
    model.config.lvr_start_id, model.config.lvr_sep_id, model.config.lvr_end_id = lvr_start_id, lvr_sep_id, lvr_end_id


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, freeze_vision_tower: bool = True, freeze_merger: bool = True, **kwargs):
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not freeze_vision_tower)
    logger.info(colored(f"Freezing vision tower: {freeze_vision_tower}", "cyan"))

    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not freeze_merger)
    logger.info(colored(f"Freezing merger: {freeze_merger}", "cyan"))

def configure_llm(model, freeze_llm: bool = False, **kwargs):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not freeze_llm)
    logger.info(colored(f"Freezing LLM Head: {freeze_llm}", "cyan"))

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not freeze_llm)
    logger.info(colored(f"Freezing LLM: {freeze_llm}", "cyan"))


def configure_latent_only(model):
    """Freeze the entire model and only train the 3 latent token embeddings.

    Uses gradient hooks on embed_tokens.weight and lm_head.weight to zero
    out all gradient rows except the lvr_start / lvr_sep / lvr_end positions.
    Must be called after set_latent_tokens() so the IDs are already set in
    model.config.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    lvr_ids = [model.config.lvr_start_id, model.config.lvr_sep_id, model.config.lvr_end_id]

    def _make_mask_hook(ids, total_rows):
        mask = torch.zeros(total_rows, dtype=torch.bool)
        for idx in ids:
            mask[idx] = True
        def hook(grad):
            masked = grad.clone()
            masked[~mask] = 0.0
            return masked
        return hook

    # embed_tokens
    embed_weight = model.model.language_model.embed_tokens.weight
    embed_weight.requires_grad = True
    embed_weight.register_hook(_make_mask_hook(lvr_ids, embed_weight.shape[0]))

    # lm_head (may share weights with embed_tokens, but register independently)
    lm_weight = model.lm_head.weight
    lm_weight.requires_grad = True
    lm_weight.register_hook(_make_mask_hook(lvr_ids, lm_weight.shape[0]))

    logger.info(colored(f"Latent-only mode: training only token IDs {lvr_ids} in embed_tokens + lm_head", "cyan"))

