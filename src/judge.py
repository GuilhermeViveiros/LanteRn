from typing import List
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import logging
logger = logging.getLogger("LantErn-Judge")

class LLMJudge:
    def __init__(self, model_id: str, device: torch.device="cuda"):
        self.model_id = model_id
        # only qwen is supported for now
        #self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model.to(device)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        #self.tokenizer.padding_side = "left"
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            padding_side="left"
        )
        self.system_prompt = (
            "You are a strict evaluation model. "
            "Given PREDICTED and GROUND TRUTH, output EXACTLY one floating-point number "
            "between 0 and 1, where 0 means completely unrelated and 1 means semantically identical. "
            "You can ignore minor differences in grammatical or function-word usage (articles, prepositions, plurals). "
            "Output only the number — no words, no explanations."
        )

    @torch.no_grad()
    def judge(self, predicted_text: List[str], ground_truth_text: List[str]) -> List[float]:
        messages = []
        # Combine messages for batch processing
        for predicted, ground_truth in zip(predicted_text, ground_truth_text):
            # create message template
            messages.append([
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"PREDICTED: {predicted.lower()}\n"
                        f"GROUND TRUTH: {ground_truth.lower()}\n\n"
                        "Output only the score. No text."
                    ),
                },
            ])
        #import ipdb; ipdb.set_trace()
        # Preparation for batch inference
        # texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages
        ]
         
        # process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            #videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        #inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        # Batch Inference
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=128,
            use_cache=True,
            #tokenizer=self.tokenizer
            tokenizer=self.processor.tokenizer
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # convert output texts to floats
        scores = np.array([float(t) for t in output_texts])
        
        return scores

if __name__ == "__main__":
    #judge = LLMJudge(model_id="Qwen/Qwen3-4B-Thinking-2507")
    #judge = LLMJudge(model_id="Qwen/Qwen3-30B-A3B-Instruct-2507")
    judge = LLMJudge(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    pt = [
        "All-In in las vegas seems is a good idea.",
        "It's a cat",
        "Its a men and women"
    ]
    gt = [
        "in las vegas All-In is a good one.",
        "It's a dog",
        "Its a men and women"
    ]

    import ipdb; ipdb.set_trace()
    answers = judge.judge(pt, gt)
    for a, b, ans in zip(pt, gt, answers):
        print(f"PT: {a} | GT: {b} | Answer: {ans:.4f}")
    # import ipdb; ipdb.set_trace()
