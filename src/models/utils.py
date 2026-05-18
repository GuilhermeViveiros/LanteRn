import os

import torch


# TODO: I could vectorize this function to avoid the loop. However since we're still in an experimental stage, the latent tokens can vary in the future, per image, so I'll leave it as is for now
def apply_latent_compression(
    self,
    latent_values,
    latent_grid_thw,
    latent_size,
):
    """
    Computes patch-averaged latent visual embeddings for training with visual reasoning tokens.

    This function is designed to process a batch of latent visual values (e.g., visual features)
    such that each sample's latent image features are compressed down to `latent_size` patch-averaged
    embeddings. These representations are intended to be used as replacements for the corresponding
    <|lvr_sep|> tokens in the language model's input sequence.

    Args:
        self: Model object with callable `get_image_features` method and `config`.
        latent_values (torch.Tensor): Batch of latent visual values (typically images or features),
          batched as required by `get_image_features`.
        latent_grid_thw (torch.LongTensor): Batch of grid dimensions for each latent input, used
          by `get_image_features` to extract visual features.
        latent_size (int): Desired number of averaged latent embeddings per sample. If -1, dynamic
          (returns same number of latent tokens as visual features).

    Returns:
        torch.Tensor: Patch-averaged latent embeddings of shape (batch_size, latent_size, hidden_dim),
          suitable for masked scattering onto <|lvr_sep|> token positions in the LLM embedding sequence.

    Notes:
        - If the number of extracted visual features does not exactly match the desired latent_size,
          features are grouped and averaged such that the final output per sample matches `latent_size`.
        - Excess features are dropped (from the end) to ensure divisibility, to minimize feature skew.
    """
    
    # ---------------------------------------------------------
    # 1) Compute latent image features once (batched)
    # ---------------------------------------------------------
    latent_image_embeds = self.get_image_features(latent_values, latent_grid_thw)
    if latent_size == -1:
        return list(latent_image_embeds)


    # ---------------------------------------------------------
    # 2) Process latent image features into exactly `latent_size` tokens
    #    (vectorizable except for variable lens, so minimal loop)
    # ---------------------------------------------------------
    processed_latents = []

    for i, le in enumerate(latent_image_embeds):
        # Count how many latent tokens exist in input
        n_features = le.shape[0]

        if latent_size != n_features:
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

def get_last_checkpoint(output_dir: str):
    """
    Get the last checkpoint from the output directory
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
    if len(checkpoints) == 0:
        return None
    return os.path.join(output_dir, max(checkpoints))

if __name__ == "__main__":
    output_dir = "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_16_lambda_0.1"
    print(get_last_checkpoint(output_dir))
