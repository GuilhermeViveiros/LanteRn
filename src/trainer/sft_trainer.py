import os
from tqdm.auto import tqdm
from typing import Callable
import torch
import wandb
from functools import partial
import numpy as np
from transformers import Trainer, AutoProcessor
from src.lantern_generate.generate import generate as lantern_generate
from transformers import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader, Dataset
from src.utils import is_rank0, get_rank, extract_mc_answer, get_world_size
import logging

logger = logging.getLogger("LantErn-Trainer")

class VisCoTestLogger(TrainerCallback):
    def __init__(
        self, 
        dataset: Dataset, 
        collate_fn: Callable, 
        processor: AutoProcessor,
        test_steps: int = 1,
        report_to: str = "wandb"
    ):
        from torch.utils.data import DataLoader
        assert test_steps > 0, "test_steps must be greater than 0"
        self.dataset = dataset
        self.processor = processor
        self.pbar = None
        self.test_steps = test_steps
        self.collate_fn = collate_fn
        self.report_to = report_to
        # if is_rank0():
        #     # get rank 0 device
        #     # 🎯 CRITICAL STEP: Use no_init_weights to isolate model loading
        #     # This prevents DeepSpeed/Accelerate's memory initialization hooks 
        #     # from interfering with the judge model's weight loading.
           
        #     self.judge = LLMJudge(
        #         model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        #         device=get_rank()
        #     )
    
    def on_step_end(self, args, state, control, metrics=None, **kwargs):
        return
        if not is_rank0():
            return
        logger.info(f"Testing at step {state.global_step}")

        if state.global_step % self.test_steps == 0: 
            with torch.no_grad():
                dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=partial(self.collate_fn, processor=self.processor))
                results = viscot_test(kwargs["model"], self.processor, dataloader, self.judge, use_gt=False)
                
                if self.report_to == "wandb":
                    wandb.log({
                        "test/avg_score": results["avg_score"],
                        "test/accuracy": results["accuracy"],
                        "test/invalid": results["invalid"],
                        "test/latent_ratio": results["latent_ratio"],
                        "step": state.global_step
                    })

class ProgressBarLossLogger(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        # deat with cases where steps > 0, resume training from a checkpoint
        if not is_rank0():
            return
        self.pbar = tqdm(total=state.max_steps, position=state.global_step, desc="Training", leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        # only rank 0 should log
        if not is_rank0():
            return

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
                "cosine_loss": cos,
                "mse_loss": mse,
                "total_loss": tot,
                "lr": kwargs["lr_scheduler"].get_last_lr()[0],
                "epoch": state.epoch,
            }, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if not is_rank0():
            return
        if self.pbar:
            self.pbar.close()

class LantErnSFTrainer(Trainer):
    def __init__(self, *args, gamma: float = 0.1, test_dataset: Dataset = None, **kwargs):
        """
        Args:
            test_dataset: Dataset containing the multiple choice test samples (custom generation test)
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        logger.info(f"Using gamma: {gamma}")
        self.mse_loss = torch.nn.MSELoss()
        self.test_dataloader = None
        if test_dataset is not None:
            self.test_dataloader = self.__initialize_test_dataloader(test_dataset)
        

    def __initialize_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Initialize the test dataloader
        """
        from torch.utils.data import DistributedSampler
        from evals.eval import collate_fn_mc
        world_size = get_world_size()
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                test_dataset, 
                #num_replicas=world_size, 
                #rank=get_rank(),
                shuffle=False,
            )
      
        # create the dataloader
        return DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=1,
            collate_fn=partial(
                collate_fn_mc,
                processor=self.processing_class
            ),
            pin_memory=True,
        )
        
    # override the evaluate function
    #  @EXPERIMENTAL: This is a custom evaluation function for multiple choice evaluation
    def __evaluate(self, eval_dataset=None, ignore_keys=None, **kwargs):
        # -------------------------
        # 1) Standard CE evaluation
        # -------------------------
        
        # skip this for now

        # metrics = super().evaluate(
        #     eval_dataset=eval_dataset,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix="eval",
        # )

        # --------------------------------
        # 2) Custom generation-based test
        # --------------------------------
        if self.test_dataloader is not None:
            test_output = self.mc_evaluation_loop(
                self.test_dataloader,
                description="Test Evaluation",
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix="test",
            )

            #metrics.update(test_output.metrics)
        if is_rank0():
            print(f"Test output: {test_output}")
            import ipdb; ipdb.set_trace()

        return metrics

    # custom evaluation function for multiple choice evaluation
    @torch.no_grad()
    def mc_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="test",
        use_lvr: bool = True,
    ):

        # TODO: I still don't know why we're using this here
        # HARDCODED FOR NOW: Check this (gviveiros)
        # if isinstance(dataloader.sampler, DistributedSampler):
        #     dataloader.sampler.set_epoch(0)
        
        # Custom generation-based evaluation
        model = self.model
        model.eval()

        all_acc = []
        all_invalid = []

        if is_rank0():
            pbar = tqdm(
                dataloader,
                desc="Evaluating MC",
                total=len(dataloader),   # per-rank length (correct)
                leave=True,
            )
        else:
            pbar = None

        for inputs, labels in dataloader:
            if pbar:
                pbar.update(1)
            inputs = inputs.to(self.args.device)
            # run generation
            with self.accelerator.unwrap_model(model) as unw_model:
                output = unw_model.generate(
                    **inputs,
                    max_new_tokens=526,
                    do_sample=False,
                    custom_generate=lantern_generate if use_lvr else None,
                    use_cache=True,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
            ]

            # decode the generated ids
            batch_decoded_output = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            print("batch_decoded_output: ", batch_decoded_output)
            
            answers = [extract_mc_answer(x) for x in batch_decoded_output]
            print("answers: ", answers)
            acc = np.array(answers) == np.array(labels)
            all_acc.append(acc.sum())

        # Distributed gathering
        all_acc = self.accelerator.gather_for_metrics(
            torch.tensor(all_acc, device=self.args.device)
        )

        metrics = {
            f"{metric_key_prefix}/accuracy": all_acc.float().mean().item(),
        }

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(all_acc),
        )

    def _prepare_mc_inputs(self, batch):
        # create messages
        messages = []
        for sample in batch:
            import ipdb; ipdb.set_trace()
            messages.append([
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in sample["image"]],
                        {"type": "text", "text": sample["question"]}
                    ]
                }
            ])
        import ipdb; ipdb.set_trace()
        # prepare the inputs
        inputs = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=inputs,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs


    # override the custom_loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        For LantErn, we also need to compute the distance between the predicted and ground truth latents
        """
        
        # return something here to skip the loss computation
        # return torch.zeros((), device=self.args.device, requires_grad=True)
        
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
        loss = ce_loss + self.gamma * mse_loss


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

    