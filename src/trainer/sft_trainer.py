import os
import re
from tqdm.auto import tqdm
from typing import Callable
import torch
import wandb
from functools import partial
from transformers import Trainer, AutoProcessor
from transformers import TrainerCallback
from torch.utils.data import DataLoader, Dataset
from src.utils import is_rank0, get_rank
import logging

logger = logging.getLogger("LantErn-Trainer")


class LatentUtilityCallback(TrainerCallback):
    """
    At each eval step, measures how much the model relies on latent visual tokens
    by sweeping 4 conditions and logging answer accuracy to wandb:

        gt     — ground-truth intermediate image as latent input
        own    — model's own generated latent tokens (default generation)
        random — random Gaussian noise instead of latent embeddings
        zeros  — zero embeddings instead of latent embeddings
    """

    def __init__(
        self,
        eval_dataset: Dataset,
        collate_fn: Callable,
        processor: AutoProcessor,
        batch_size: int = 8,
        max_eval_samples: int = 80,
        seed: int = 42,
    ):
        import random as _random
        rng = _random.Random(seed)
        n = min(max_eval_samples, len(eval_dataset))
        indices = rng.sample(range(len(eval_dataset)), n)
        from torch.utils.data import Subset
        self.eval_dataset = Subset(eval_dataset, sorted(indices))
        self.collate_fn   = collate_fn
        self.processor    = processor
        self.batch_size   = batch_size

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Pull the letter out of <answer>...</answer> or last (a/b/c/d) token."""
        m = re.search(r'<answer>\s*([a-d])\s*</answer>', text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        m = re.search(r'\b([a-d])\b', text[::-1], re.IGNORECASE)
        return m.group(1).lower() if m else ""

    @torch.no_grad()
    def _run_condition(self, model, dataloader, use_gt: bool, perturbation, desc: str = "", n_log: int = 8):
        from evals import run_batch_inference
        lvr_start_token = "<|lvr_start|>"
        correct   = 0
        lvr_used  = 0
        total     = 0
        log_preds = []   # decoded strings for wandb table (first n_log)
        log_gts   = []   # ground truth labels for wandb table
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            out = run_batch_inference(
                model, inputs,
                use_lvr=True,
                use_gt=use_gt,
                perturbation=perturbation,
            )
            seqs = out.sequences if hasattr(out, "sequences") else out
            decoded = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=False)
            for pred, gt in zip(decoded, labels):
                pred_ans = self._extract_answer(pred)
                correct  += int(pred_ans == gt.strip().lower())
                lvr_used += int(lvr_start_token in pred)
                total    += 1
                if len(log_preds) < n_log:
                    log_preds.append(pred[:1000])  # truncate long traces
                    log_gts.append(gt.strip())
        if total == 0:
            return 0.0, 0.0, [], []
        return correct / total, lvr_used / total, log_preds, log_gts

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        _dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank  = torch.distributed.get_rank() if _dist else 0

        # Each condition is assigned to one rank (0-3); ranks >= 4 sit idle.
        conditions = [
            ("latent_utility/gt",     dict(use_gt=True,  perturbation=None)),
            ("latent_utility/own",    dict(use_gt=False, perturbation=None)),
            ("latent_utility/random", dict(use_gt=False, perturbation="random")),
            ("latent_utility/zeros",  dict(use_gt=False, perturbation="zeros")),
        ]

        _C = "\033[96m"   # cyan
        _R = "\033[0m"

        acc      = 0.0
        lvr_rate = 0.0
        log_preds, log_gts = [], []
        if rank < len(conditions):
            name, cond_kwargs = conditions[rank]
            label = name.split("/")[-1]
            print(f"{_C}[LatentUtility] rank {rank} → evaluating: {label}{_R}", flush=True)

            unwrapped = model.module if hasattr(model, "module") else model
            orig_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"
            dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                collate_fn=partial(self.collate_fn, processor=self.processor),
            )
            unwrapped.eval()
            acc, lvr_rate, log_preds, log_gts = self._run_condition(
                unwrapped, dataloader, **cond_kwargs, desc=f"rank {rank} [{label}]"
            )
            self.processor.tokenizer.padding_side = orig_padding_side
            print(f"{_C}[LatentUtility] rank {rank} | {label}: acc={acc:.3f}  lvr_rate={lvr_rate:.3f}{_R}", flush=True)

        # Gather scalar metrics [acc, lvr_rate] per rank → rank 0.
        if _dist:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            metrics_t   = torch.tensor([acc, lvr_rate], dtype=torch.float32, device=device)
            world_size  = torch.distributed.get_world_size()
            gather_list = [torch.zeros(2, dtype=torch.float32, device=device)
                           for _ in range(world_size)] if rank == 0 else None
            torch.distributed.gather(metrics_t, gather_list, dst=0)
            # Gather decoded strings via gather_object (CPU, no size constraint)
            gen_obj = {"preds": log_preds, "gts": log_gts}
            gen_gather = [None] * world_size if rank == 0 else None
            torch.distributed.gather_object(gen_obj, gen_gather, dst=0)
            torch.distributed.barrier()
        else:
            gather_list = [torch.tensor([acc, lvr_rate])]
            gen_gather  = [{"preds": log_preds, "gts": log_gts}]

         # Send to wandb
        if wandb.run:
            if rank == 0:
                results = {}
                for i, (name, _) in enumerate(conditions):
                    label = name.split("/")[-1]
                    results[name]                               = gather_list[i][0].item()
                    results[f"latent_utility/{label}_lvr_rate"] = gather_list[i][1].item()
                print(f"{_C}[LatentUtility] step {state.global_step} → {results}{_R}", flush=True)

                wandb.log(results, step=state.global_step)
                # Build generation table from the 4 conditions (ranks 0-3)
                cond_preds = {conditions[i][0].split("/")[-1]: gen_gather[i]["preds"]
                              for i in range(len(conditions))}
                gts = gen_gather[0]["gts"]  # same across all conditions
                n   = min(len(gts), min(len(v) for v in cond_preds.values()))
                if n > 0:
                    table = wandb.Table(columns=["sample", "gt", "pred_gt", "pred_own", "pred_random", "pred_zeros"])
                    for j in range(n):
                        table.add_data(
                            j, gts[j],
                            cond_preds["gt"][j],
                            cond_preds["own"][j],
                            cond_preds["random"][j],
                            cond_preds["zeros"][j],
                        )
                    wandb.log({"latent_utility/generations": table}, step=state.global_step)

class GenerationAccuracyCallback(TrainerCallback):
    """
    Simple generation-based accuracy eval for NTP (or any model without LVR).
    Splits samples across all ranks, gathers results to rank 0, logs to wandb.
    """

    def __init__(
        self,
        eval_dataset: Dataset,
        collate_fn: Callable,
        processor: AutoProcessor,
        batch_size: int = 8,
        max_eval_samples: int = 300,
        seed: int = 42,
    ):
        import random as _random
        rng = _random.Random(seed)
        n = min(max_eval_samples, len(eval_dataset))
        indices = rng.sample(range(len(eval_dataset)), n)
        from torch.utils.data import Subset
        self.eval_dataset = Subset(eval_dataset, sorted(indices))
        self.collate_fn   = collate_fn
        self.processor    = processor
        self.batch_size   = batch_size

    @staticmethod
    def _extract_answer(text: str) -> str:
        m = re.search(r'<answer>\s*([a-d])\s*</answer>', text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        m = re.search(r'\b([a-d])\b', text[::-1], re.IGNORECASE)
        return m.group(1).lower() if m else ""

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        _dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank       = torch.distributed.get_rank()       if _dist else 0
        world_size = torch.distributed.get_world_size() if _dist else 1

        _C = "\033[96m"; _R = "\033[0m"

        # Split samples across ranks
        from torch.utils.data import Subset
        all_indices = list(range(len(self.eval_dataset)))
        rank_indices = all_indices[rank::world_size]
        rank_dataset = Subset(self.eval_dataset, rank_indices)

        unwrapped = model.module if hasattr(model, "module") else model
        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        dataloader = DataLoader(
            rank_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(self.collate_fn, processor=self.processor),
        )
        unwrapped.eval()

        from evals import run_batch_inference
        correct = 0
        total   = 0
        log_preds, log_gts = [], []
        for inputs, labels in tqdm(dataloader, desc=f"rank {rank} [gen_acc]", leave=False):
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            out = run_batch_inference(model, inputs, use_lvr=False, use_gt=False)
            seqs    = out.sequences if hasattr(out, "sequences") else out
            decoded = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=False)
            for pred, gt in zip(decoded, labels):
                pred_ans = self._extract_answer(pred)
                correct += int(pred_ans == gt.strip().lower())
                total   += 1
                if len(log_preds) < 8:
                    log_preds.append(pred[:1000])
                    log_gts.append(gt.strip())

        self.processor.tokenizer.padding_side = orig_padding_side
        print(f"{_C}[GenAcc] rank {rank}: {correct}/{total} = {correct/max(total,1):.3f}{_R}", flush=True)

        # Gather across ranks
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        metrics_t   = torch.tensor([correct, total], dtype=torch.float32, device=device)
        gather_list = [torch.zeros(2, dtype=torch.float32, device=device)
                       for _ in range(world_size)] if rank == 0 else None
        if _dist:
            torch.distributed.gather(metrics_t, gather_list, dst=0)
            gen_obj    = {"preds": log_preds, "gts": log_gts}
            gen_gather = [None] * world_size if rank == 0 else None
            torch.distributed.gather_object(gen_obj, gen_gather, dst=0)
            torch.distributed.barrier()
        else:
            gather_list = [metrics_t]
            gen_gather  = [{"preds": log_preds, "gts": log_gts}]

        if rank == 0:
            total_correct = sum(g[0].item() for g in gather_list)
            total_total   = sum(g[1].item() for g in gather_list)
            acc = total_correct / max(total_total, 1)
            print(f"{_C}[GenAcc] step {state.global_step} → acc={acc:.3f} ({int(total_correct)}/{int(total_total)}){_R}", flush=True)
            wandb.log({"eval/generation_acc": acc}, step=state.global_step)

            all_preds = [p for g in gen_gather for p in g["preds"]]
            all_gts   = [g for obj in gen_gather for g in obj["gts"]]
            n = min(len(all_gts), len(all_preds))
            if n > 0:
                table = wandb.Table(columns=["sample", "gt", "prediction"])
                for j in range(n):
                    table.add_data(j, all_gts[j], all_preds[j])
                wandb.log({"eval/generations": table}, step=state.global_step)


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
    def __init__(self, *args, gamma: float = 0.1, latent_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.latent_only = latent_only
        # only rank0
        if is_rank0():
            logger.info(f"Using gamma: {gamma}, latent_only: {latent_only}")
        self.mse_loss = torch.nn.MSELoss()

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

        # In latent-only mode, only CE loss is needed — no MSE supervision. 
        # Ablation study -> training the latent weights only
        if self.latent_only:
            return (ce_loss, outputs) if return_outputs else ce_loss

        hidden_states = outputs.hidden_states
        input_embeddings = outputs.inputs_embeds
        input_ids = inputs["input_ids"]

        # get idx where token is <|lvr_sep|> / <|lvr_start|>
        lvr_sep_mask   = (input_ids == self.model.config.lvr_sep_id)
        lvr_start_mask = (input_ids == self.model.config.lvr_start_id)
        if lvr_sep_mask.sum() == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Shifted MSE — k pairs for k GT visual embeds (latent_size = k):
        #
        #   h0 = H[lvr_start_pos] ≈ v0   ← full prefix up to <lvr_start>, hidden state there
        #   h1 = H[sep_0_pos]     ≈ v1   ← step 1: input was h0
        #   h2 = H[sep_1_pos]     ≈ v2
        #   h3 = H[sep_2_pos]     ≈ v3   (k=4 example)
        #   h4 = H[sep_3_pos]     → CE(<lvr_end>) only, no MSE target
        #
        #   pred = [lvr_start_pos, sep_0, ..., sep_{k-2}]  (k positions)
        #   gt   = [sep_0,         sep_1, ..., sep_{k-1}]  (v0..v_{k-1})
        pred_list, gt_list = [], []
        for b in range(input_ids.shape[0]):
            start_pos = lvr_start_mask[b].nonzero(as_tuple=False).squeeze(-1)  # (1,)
            sep_pos   = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)    # (k,)
            assert start_pos.numel() > 0 and sep_pos.numel() > 0, \
                f"Sample {b} has no latent tokens — every sample must have <lvr_start> and <lvr_sep>"
            pred_idxs = torch.cat([start_pos, sep_pos[:-1]])  # [lvr_start, sep_0..sep_{k-2}]
            pred_list.append(hidden_states[b, pred_idxs])     # k hidden states
            gt_list.append(input_embeddings[b, sep_pos])      # v0..v_{k-1}

        if not pred_list:
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred_embeddings = torch.cat(pred_list, dim=0)
        gt_embeddings   = torch.cat(gt_list,   dim=0)

        # compute the distance between the predicted and ground truth latents
        sim_loss = 1-torch.nn.functional.cosine_similarity(pred_embeddings, gt_embeddings).mean()
        mse_loss = self.mse_loss(pred_embeddings, gt_embeddings)
        
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

    