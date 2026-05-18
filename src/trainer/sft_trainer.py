import logging
import re
from functools import partial
from typing import Callable

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, Trainer, TrainerCallback

from src.utils import is_rank0

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
        max_eval_samples: int = 1000,
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

    @staticmethod
    def _collapse_vision_tokens(text: str) -> str:
        """Replace <|vision_start|>...<|vision_end|> blocks with <img>."""
        return re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '<img>', text, flags=re.DOTALL)

    @torch.no_grad()
    def _run_condition(self, model, dataloader, use_gt: bool, perturbation, desc: str = "", n_log: int = 32):
        from evals import run_batch_inference
        lvr_start_token = "<|lvr_start|>"
        correct    = 0
        lvr_used   = 0
        total      = 0
        log_prompts = []
        log_preds   = []
        log_gts     = []
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            prompt_len = inputs["input_ids"].shape[1]
            out = run_batch_inference(
                model, inputs,
                use_lvr=True,
                use_gt=use_gt,
                perturbation=perturbation,
            )
            seqs = out.sequences if hasattr(out, "sequences") else out
            prompts   = self.processor.tokenizer.batch_decode(seqs[:, :prompt_len],  skip_special_tokens=False)
            generated = self.processor.tokenizer.batch_decode(seqs[:, prompt_len:],  skip_special_tokens=False)
            for prompt, gen, gt in zip(prompts, generated, labels):
                pred_ans = self._extract_answer(gen)
                correct  += int(pred_ans == gt.strip().lower())
                lvr_used += int(lvr_start_token in gen)
                total    += 1
                if len(log_preds) < n_log:  # noqa: SIM102
                    log_prompts.append(self._collapse_vision_tokens(prompt))
                    log_preds.append(gen)
                    log_gts.append(gt.strip())
        if total == 0:
            return 0.0, 0.0, [], [], []
        return correct / total, lvr_used / total, log_prompts, log_preds, log_gts

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        _dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank  = torch.distributed.get_rank() if _dist else 0

        # Each condition is assigned to one rank; ranks >= len(conditions) sit idle.
        conditions = [
            ("latent_utility/gt",     dict(use_gt=True,  perturbation=None)),
            ("latent_utility/own",    dict(use_gt=False, perturbation=None)),
            ("latent_utility/zeros",  dict(use_gt=True,  perturbation="zeros")),
            ("latent_utility/random", dict(use_gt=True,  perturbation="random")),
        ]

        _C = "\033[96m"   # cyan
        _R = "\033[0m"

        acc      = 0.0
        lvr_rate = 0.0
        log_prompts, log_preds, log_gts = [], [], []
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
            acc, lvr_rate, log_prompts, log_preds, log_gts = self._run_condition(
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
            gen_obj = {"prompts": log_prompts, "preds": log_preds, "gts": log_gts}
            gen_gather = [None] * world_size if rank == 0 else None
            torch.distributed.gather_object(gen_obj, gen_gather, dst=0)
            torch.distributed.barrier()
        else:
            gather_list = [torch.tensor([acc, lvr_rate])]
            gen_gather  = [{"prompts": log_prompts, "preds": log_preds, "gts": log_gts}]

         # Send to wandb
        if wandb.run:
            if rank == 0:
                results = {}
                for i, (name, _) in enumerate(conditions):
                    label = name.split("/")[-1]
                    results[name]                               = gather_list[i][0].item()
                    results[f"latent_utility/{label}_lvr_rate"] = gather_list[i][1].item()
                print(f"{_C}[LatentUtility] step {state.global_step} → {results}{_R}", flush=True)

                results["train/global_step"] = state.global_step
                wandb.log(results)

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

        _C = "\033[96m"
        _R = "\033[0m"

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
            prompt_len = inputs["input_ids"].shape[1]
            out = run_batch_inference(model, inputs, use_lvr=False, use_gt=False,
                                      output_attentions=False, return_dict=True)
            seqs    = out.sequences if hasattr(out, "sequences") else out
            decoded = self.processor.tokenizer.batch_decode(
                seqs[:, prompt_len:], skip_special_tokens=False
            )
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
        assert test_steps > 0, "test_steps must be greater than 0"
        self.dataset = dataset
        self.processor = processor
        self.pbar = None
        self.test_steps = test_steps
        self.collate_fn = collate_fn
        self.report_to = report_to

    def on_step_end(self, args, state, control, metrics=None, **kwargs):
        pass

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

        if self.pbar:
            ce  = logs.get("ce_loss")
            tot = logs.get("total_loss")
            mse = logs.get("mse_loss")
            nce = logs.get("infonce_loss")
            cos = logs.get("cosine_loss")
            if ce is not None and tot is not None:
                postfix = {"ce": f"{ce:.4f}", "mse": f"{mse:.4f}", "loss": f"{tot:.4f}"}
                if nce is not None:
                    postfix["nce"] = f"{nce:.4f}"
                else:
                    postfix["cos"] = f"{cos:.4f}"
                self.pbar.set_postfix(postfix)
            elif tot is not None:
                self.pbar.set_postfix({"loss": f"{tot:.4f}"})
            self.pbar.update(1)

        # keep only last 100 entries
        if len(state.log_history) > 100:
            state.log_history.pop(0)

    def on_train_end(self, args, state, control, **kwargs):
        if not is_rank0():
            return
        if self.pbar:
            self.pbar.close()

class LantErnSFTrainer(Trainer):
    def __init__(self, *args, gamma: float = 0.1, latent_only: bool = False, use_lvr: bool = True, latent_loss_type: str = "mse", temperature: float = 0.07, scheduled_sampling_prob: float = 0.0, scheduled_sampling_warmup: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.latent_only = latent_only
        self.use_lvr = use_lvr
        self.latent_loss_type = latent_loss_type
        self.temperature = temperature
        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.scheduled_sampling_warmup = scheduled_sampling_warmup
        # only rank0
        if is_rank0():
            logger.info(f"Using gamma: {gamma}, latent_only: {latent_only}, latent_loss_type: {latent_loss_type}, temperature: {temperature}, scheduled_sampling_prob: {scheduled_sampling_prob}, scheduled_sampling_warmup: {scheduled_sampling_warmup}")
        self.mse_loss = torch.nn.MSELoss()

    def _get_train_sampler(self, dataset=None):
        from torch.utils.data import SequentialSampler

        from src.datasets.family_sampler import FamilyGroupedDataset
        # FamilyGroupedDataset pre-orders indices by family chunk — must use
        # SequentialSampler to preserve that ordering.  For all other datasets
        # fall back to the default (RandomSampler / distributed sampler).
        ds = dataset if dataset is not None else self.train_dataset
        if isinstance(ds, FamilyGroupedDataset):
            return SequentialSampler(ds)
        return super()._get_train_sampler(dataset)

    @staticmethod
    def infonce_loss(pred: torch.Tensor, gt: torch.Tensor, temperature: float = 0.07,
                     hard_negative_mask: torch.Tensor = None) -> torch.Tensor:
        """Bidirectional NT-Xent InfoNCE loss. pred/gt: [N, D].
        hard_negative_mask: [N, N] bool — True for same-shape_C pairs (i!=j).
        Hard negative logits are upweighted by log(2) to force tighter discrimination.
        """
        pred_n = torch.nn.functional.normalize(pred, dim=-1)
        gt_n   = torch.nn.functional.normalize(gt,   dim=-1)
        logits = torch.matmul(pred_n, gt_n.t()) / temperature  # [N, N]
        if hard_negative_mask is not None:
            import math
            logits = logits + hard_negative_mask.float() * math.log(2.0)
        labels = torch.arange(pred.size(0), device=pred.device)
        return (torch.nn.functional.cross_entropy(logits, labels) +
                torch.nn.functional.cross_entropy(logits.t(), labels)) / 2.0

    # override the custom_loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        For LantErn, we also need to compute the distance between the predicted and ground truth latents
        """

        input_ids = inputs["input_ids"]

        # NTP mode: no latent tokens, pure CE loss.
        if not self.use_lvr:
            outputs = model(**inputs, return_dict=True)
            ce_loss = outputs.loss
            if wandb.run and is_rank0():
                lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0.0
                wandb.log({"ce_loss": ce_loss.item(), "total_loss": ce_loss.item(), "lr": lr, "epoch": self.state.epoch})
            return (ce_loss, outputs) if return_outputs else ce_loss

        # In latent-only mode, only CE loss is needed — no MSE supervision.
        if self.latent_only:
            outputs = model(**inputs, return_dict=True)
            ce_loss = outputs.loss
            return (ce_loss, outputs) if return_outputs else ce_loss

        # get idx where token is <|lvr_sep|> / <|lvr_start|>
        lvr_sep_mask   = (input_ids == self.model.config.lvr_sep_id)
        lvr_start_mask = (input_ids == self.model.config.lvr_start_id)

        # ── Compute n_replaced BEFORE any forward pass ──────────────────────────
        n_replaced = 0
        if self.scheduled_sampling_prob > 0 and self.state.max_steps > 0:
            progress = self.state.global_step / self.state.max_steps
            warmup   = self.scheduled_sampling_warmup
            if is_rank0() and self.state.global_step % 10 == 0:
                logger.info(f"[SchedSampling] step={self.state.global_step}/{self.state.max_steps} "
                            f"progress={progress:.3f} warmup={warmup}")
            if progress >= warmup:
                sched_progress = (progress - warmup) / max(1.0 - warmup, 1e-8)
                latent_size    = self.model.config.latent_size
                n_replaced     = round(sched_progress * self.scheduled_sampling_prob * latent_size)
                n_replaced     = min(n_replaced, latent_size)
                if is_rank0() and self.state.global_step % 10 == 0:
                    logger.info(f"[SchedSampling] sched_progress={sched_progress:.3f} "
                                f"n_replaced={n_replaced}/{latent_size}")

        _PIXEL_KEYS = ('input_ids', 'pixel_values', 'image_grid_thw',
                       'pixel_values_videos', 'video_grid_thw',
                       'latent_values', 'latent_grid_thw',
                       'latent_mask_out', 'answer_positions')

        if n_replaced > 0:
            # ── Two-pass scheduled sampling — SINGLE backward (no ZeRO conflict) ──
            # Pass 1: no_grad — only used to build self-predicted embeddings
            with torch.no_grad():
                ref_outputs     = model(**inputs, return_dict=True)
                ref_hidden      = ref_outputs.hidden_states   # [B, T, D]
                gt_input_embeds = ref_outputs.inputs_embeds   # [B, T, D]

            # Collect hidden states at [lvr_start, sep_0..sep_{k-2}] for each sample
            ref_pred_list = []
            for b in range(input_ids.shape[0]):
                start_pos = lvr_start_mask[b].nonzero(as_tuple=False).squeeze(-1)
                sep_pos   = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)
                pred_idxs = torch.cat([start_pos, sep_pos[:-1]])
                ref_pred_list.append(ref_hidden[b, pred_idxs])

            # Replace last n sep embeddings with model's own predictions
            new_embeds = gt_input_embeds.clone()
            for b in range(input_ids.shape[0]):
                sep_pos = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)
                new_embeds[b, sep_pos[-n_replaced:]] = ref_pred_list[b][-n_replaced:]

            # gt_list: GT visual embeddings from the original input (detached — no grad needed)
            gt_list = [gt_input_embeds[b, lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)].detach()
                       for b in range(input_ids.shape[0])]

            # Pass 2: with grad — the ONLY backward pass
            second_inputs = {k: v for k, v in inputs.items() if k not in _PIXEL_KEYS}
            second_inputs['inputs_embeds'] = new_embeds
            outputs      = model(**second_inputs, return_dict=True)
            ce_loss      = outputs.loss
            hidden_states    = outputs.hidden_states
            input_embeddings = new_embeds

            # pred_list from second forward — trains latent production under self-context
            pred_list = []
            for b in range(input_ids.shape[0]):
                start_pos = lvr_start_mask[b].nonzero(as_tuple=False).squeeze(-1)
                sep_pos   = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)
                pred_idxs = torch.cat([start_pos, sep_pos[:-1]])
                pred_list.append(hidden_states[b, pred_idxs])

            if is_rank0() and self.state.global_step % 10 == 0:
                logger.info(f"[SchedSampling] second forward done — ce_loss_ss={ce_loss.item():.4f}")

        else:
            # ── Standard single forward ──────────────────────────────────────────
            outputs          = model(**inputs, return_dict=True)
            ce_loss          = outputs.loss
            hidden_states    = outputs.hidden_states
            input_embeddings = outputs.inputs_embeds

            # Shifted MSE — k pairs for k GT visual embeds (latent_size = k):
            #   pred = [lvr_start_pos, sep_0, ..., sep_{k-2}]  (k positions)
            #   gt   = [sep_0,         sep_1, ..., sep_{k-1}]  (v0..v_{k-1})
            pred_list, gt_list = [], []
            for b in range(input_ids.shape[0]):
                start_pos = lvr_start_mask[b].nonzero(as_tuple=False).squeeze(-1)
                sep_pos   = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if start_pos.numel() == 0 or sep_pos.numel() == 0:
                    raise ValueError(f"Sample {b} has no latent tokens — every sample must have <lvr_start> and <lvr_sep>")
                pred_idxs = torch.cat([start_pos, sep_pos[:-1]])
                pred_list.append(hidden_states[b, pred_idxs])
                gt_list.append(input_embeddings[b, sep_pos])

        if not pred_list:
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred_embeddings = torch.cat(pred_list, dim=0)
        gt_embeddings   = torch.cat(gt_list,   dim=0)

        # Per-token (flat) — for MSE logging and token-level cosine monitoring
        sim_loss = 1 - torch.nn.functional.cosine_similarity(pred_embeddings, gt_embeddings).mean()
        mse_loss = self.mse_loss(pred_embeddings, gt_embeddings)

        # Block-level: flatten latent_size tokens per sample → [B, latent_size * D]
        pred_block = torch.stack([p.flatten() for p in pred_list])
        gt_block   = torch.stack([g.flatten() for g in gt_list])

        # Build hard-negative mask from shape_name_ids if available (same shape, different sample).
        # shape_name_ids is an int64 tensor (zlib.adler32 of the name); -1 means unknown.
        hard_negative_mask = None
        shape_name_ids = inputs.get("shape_name_ids")
        if shape_name_ids is not None and self.latent_loss_type in ("infonce", "mse+infonce"):
            ids  = shape_name_ids.to(pred_block.device)
            mask = (ids.unsqueeze(0) == ids.unsqueeze(1))   # [N, N] — same shape name
            mask.fill_diagonal_(False)                       # exclude self-pairs
            valid = ids >= 0
            mask &= valid.unsqueeze(0) & valid.unsqueeze(1) # exclude unknown (-1)
            if mask.any():
                hard_negative_mask = mask

        nce_loss = None
        if self.latent_loss_type == "mse":
            latent_loss = mse_loss
        elif self.latent_loss_type == "infonce":
            nce_loss = self.infonce_loss(pred_block, gt_block, self.temperature, hard_negative_mask)
            latent_loss = nce_loss
        elif self.latent_loss_type == "cosine":
            latent_loss = 1 - torch.nn.functional.cosine_similarity(pred_block, gt_block).mean()
        elif self.latent_loss_type == "mse+infonce":
            nce_loss = self.infonce_loss(pred_block, gt_block, self.temperature, hard_negative_mask)
            latent_loss = mse_loss + nce_loss
        else:
            raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type!r}")

        # compute the total loss
        loss = ce_loss + self.gamma * latent_loss


        # HF Trainer logging (no prog_bar)
        if hasattr(self, "state") and hasattr(self.state, "log_history"):
            entry = {
                "ce_loss": ce_loss.item(),
                "mse_loss": mse_loss.item(),
                "cosine_loss": sim_loss.item(),
                "latent_loss": latent_loss.item(),
                "total_loss": loss.item(),
            }
            if nce_loss is not None:
                entry["infonce_loss"] = nce_loss.item()
            self.state.log_history.append(entry)

        if wandb.run and is_rank0():
            lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0.0
            log_dict = {
                "ce_loss": ce_loss.item(),
                "mse_loss": mse_loss.item(),
                "cosine_loss": sim_loss.item(),
                "latent_loss": latent_loss.item(),
                "total_loss": loss.item(),
                "lr": lr,
                "epoch": self.state.epoch,
                "scheduled_sampling/n_replaced": n_replaced,
                "scheduled_sampling/sched_progress": (max(0, (self.state.global_step / self.state.max_steps) - self.scheduled_sampling_warmup) / max(1.0 - self.scheduled_sampling_warmup, 1e-8)) if self.state.max_steps > 0 else 0.0,
            }
            if nce_loss is not None:
                log_dict["infonce_loss"] = nce_loss.item()
            if hard_negative_mask is not None:
                N = hard_negative_mask.shape[0]
                log_dict["hard_neg_fraction"] = hard_negative_mask.sum().item() / max(N * (N - 1), 1)
            wandb.log(log_dict)

        return (loss, outputs) if return_outputs else loss



