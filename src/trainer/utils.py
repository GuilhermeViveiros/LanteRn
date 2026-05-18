from itertools import chain

import torch


def prepare_latent_embeds(latent_embeds: list[torch.Tensor]) -> torch.Tensor:
    latent_embeds = [embed for embed in latent_embeds if len(embed) > 0] # ignore empty lists
    latent_embeds = list(chain.from_iterable(latent_embeds)) # nested list to single list
    latent_embeds = torch.stack(latent_embeds, dim=0).to(dtype=torch.bfloat16)
    return latent_embeds
