from tqdm.auto import tqdm
import torch
import wandb
from transformers import Trainer
from transformers import TrainerCallback

class ProgressBarLossLogger(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if len(state.log_history) > 0 else {}
        
        if logs and self.pbar:
            ce = logs.get("ce_loss")
            cos = logs.get("cosine_loss")
            tot = logs.get("total_loss")
            mse = logs.get("mse_loss")
            if ce is not None and cos is not None and tot is not None and mse is not None:
                self.pbar.set_postfix({
                    "ce": f"{ce:.4f}",
                    "lvr": f"{cos:.4f}",
                    "loss": f"{tot:.4f}"
                })
            self.pbar.update(1)
        
        # keep only last 100 entries
        if len(state.log_history) > 100:
            state.log_history.pop(0)

        # Send to wandb
        if wandb.run:
            wandb.log({
                "ce_loss": ce,
                "lvr_loss": cos,
                "mse_loss": mse,
                "total_loss": tot,
                "lr": kwargs["lr_scheduler"].get_last_lr()[0],
                "epoch": state.epoch,
            }, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

class LantErnSFTrainer(Trainer):
    def __init__(self, *args, gamma: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    # override the custom_loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        For LantErn, we also need to compute the distance between the predicted and ground truth latents
        """
        outputs = model(
            **inputs,
            return_dict=True
        )
        ce_loss = outputs.loss
        hidden_states = outputs.hidden_states
        input_embeddings = outputs.inputs_embeds
        latent_mask_out = inputs["latent_mask_out"]
        # extract the latent values for the visual reasoning process
        input_ids = inputs["input_ids"]
        # get idx where token is <|lvr_sep|>
        lvr_sep_idx = (input_ids == self.model.config.lvr_sep_id)
        if lvr_sep_idx.sum() == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss
        # get predicted latent visuals
        pred_embeddings = hidden_states[lvr_sep_idx]
        # shift the predicted embeddings by 1
        pred_embeddings = pred_embeddings[1:]
        # get ground truth latent visuals
        gt_embeddings = input_embeddings[lvr_sep_idx]
        # compute the distance between the predicted and ground truth latents
        sim_loss = 1-torch.nn.functional.cosine_similarity(gt_embeddings, pred_embeddings).mean()
        mse_loss = torch.nn.functional.mse_loss(gt_embeddings, pred_embeddings)
        
        # compute the total loss
        loss = self.gamma * ce_loss + mse_loss


        # HF Trainer logging (no prog_bar)
        # ✅ Instead of self.log(...), store it silently:
        if hasattr(self, "state") and hasattr(self.state, "log_history"):
            self.state.log_history.append({
                "ce_loss": ce_loss.item(),
                "mse_loss": mse_loss.item(),
                "cosine_loss": sim_loss.item(),
                "total_loss": loss.item(),
            })

        return (loss, outputs) if return_outputs else loss

    