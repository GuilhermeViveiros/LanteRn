import torch

def apply_latent_compression(
    self,
    input_ids,
    latent_values,
    latent_grid_thw,
):
    """
    Vectorized replacement of <lvr_sep> token embeddings with latent patch-averaged embeddings.
    """

    latent_size = self.model.config.latent_size  # fixed number of latent reasoning tokens

    # ---------------------------------------------------------
    # 1) Compute latent image features once (batched)
    # ---------------------------------------------------------
    latent_image_embeds = self.get_image_features(latent_values, latent_grid_thw)
    
    # ---------------------------------------------------------
    # 2) Process latent image features into exactly `latent_size` tokens
    #    (vectorizable except for variable lens, so minimal loop)
    # ---------------------------------------------------------
    processed_latents = []

    for i, le in enumerate(latent_image_embeds):
        # Count how many latent tokens exist in input
        n_latent_tokens = (input_ids[i] == self.config.lvr_sep_id).sum().item()
        n_features = le.shape[0]

        if n_latent_tokens != n_features:
            leave_out = n_features % latent_size # Make divisible
            if leave_out > 0:
                le = le[:-leave_out, :]

            group_size = le.shape[0] // latent_size

            le = (
                le.view(latent_size, group_size, self.config.hidden_size)
                  .mean(dim=1)  # -> (latent_size, hidden)
            )
        processed_latents.append(le)

    # Stack final latent embeddings:   (batch, latent_size, hidden)
    latent_avg_embeds = torch.stack(processed_latents, dim=0)
    
    return latent_avg_embeds
