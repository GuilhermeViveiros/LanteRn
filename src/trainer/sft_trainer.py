from tqdm.auto import tqdm
from typing import Callable
import torch
import wandb
import subprocess
from functools import partial
from transformers import Trainer, AutoProcessor
from transformers import TrainerCallback
from torch.utils.data import DataLoader, Dataset
from src.judge import LLMJudge
from src.test import viscot_test
import subprocess

class VisCoTestLogger(TrainerCallback):
    def __init__(
        self, 
        dataset: Dataset, 
        collate_fn: Callable, 
        processor: AutoProcessor,
        test_steps: int = 1
    ):
        from torch.utils.data import DataLoader
        assert test_steps > 0, "test_steps must be greater than 0"
        self.dataset = dataset
        self.processor = processor
        self.pbar = None
        self.test_steps = test_steps
        self.collate_fn = collate_fn
        self.judge = LLMJudge(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    def on_step_end(self, args, state, control, metrics=None, **kwargs):
        # calling a subprocess to run the test script
        # avoid interruping the training with jibberish stuff
        # subprocess.run(["bash", "scripts/eval_viscot.sh", args.model_ref, args.data_path])
        if state.global_step % self.test_steps == 0:

            # get the model ckpt at the current step
            # TODO: implement this
            
            #dataloader = DataLoader(self.dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=self.collate_fn)
            dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=partial(self.collate_fn, processor=self.processor))
            
            viscot_test(kwargs["model"], self.processor, dataloader, self.judge)

class EvalLossLogger(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics comes from trainer.evaluate()
        # metrics already contains 'eval_loss' if compute_loss returns a scalar

        if metrics is None:
            return

        # Show custom losses if present
        ce = metrics.get("ce_loss")
        cos = metrics.get("cosine_loss")
        mse = metrics.get("mse_loss")
        tot = metrics.get("total_loss")

        # TQDM bar
        if self.pbar is None:
            self.pbar = tqdm(total=1, desc="Evaluation", leave=True)
        self.pbar.set_postfix({
            "ce": f"{ce:.4f}" if ce is not None else "-",
            "lvr": f"{cos:.4f}" if cos is not None else "-",
            "mse": f"{mse:.4f}" if mse is not None else "-",
            "loss": f"{tot:.4f}" if tot is not None else "-"
        })
        self.pbar.update(1)

        # log to wandb
        if wandb.run:
            log_dict = {k: v for k, v in metrics.items() if v is not None}
            log_dict["step"] = state.global_step
            wandb.log(log_dict)

        # close the bar after eval
        self.pbar.close()
        self.pbar = None

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
                    "mse": f"{mse:.4f}",
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
        self.mse_loss = torch.nn.MSELoss()

    # override the custom_loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        For LantErn, we also need to compute the distance between the predicted and ground truth latents
        """
        
        print(f"inputs: {inputs['input_ids'].shape}")
        outputs = model(
            **inputs,
            return_dict=True
        )
        ce_loss = outputs.loss
        hidden_states = outputs.hidden_states
        input_embeddings = outputs.inputs_embeds
        latent_mask_out = inputs["latent_mask_out"]
        input_ids = inputs["input_ids"]
        
        # get idx where token is <|lvr_sep|>
        lvr_sep_idx = (input_ids == self.model.config.lvr_sep_id)
        if lvr_sep_idx.sum() == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        shift_image_mask = latent_mask_out[..., -(hidden_states.shape[1]-1):]
        shift_pred_embeddings = hidden_states[...,:-1,:][shift_image_mask != 0]
        
        # get ground truth latent visuals
        shift_gt_embeddings = input_embeddings[:,1:,:][shift_image_mask != 0]

        # compute the distance between the predicted and ground truth latents
        sim_loss = 1-torch.nn.functional.cosine_similarity(shift_pred_embeddings, shift_gt_embeddings).mean()
        mse_loss = self.mse_loss(shift_pred_embeddings, shift_gt_embeddings)
        
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

    