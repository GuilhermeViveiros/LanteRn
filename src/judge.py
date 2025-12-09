import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

class LLMJudge:
    def __init__(self, model_id: str):
        self.model_id = model_id
        # only qwen is supported for now
        # self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.system_prompt = (
            "You are a strict evaluation model. "
            "Given PREDICTED and GROUND TRUTH, output EXACTLY one floating-point number "
            "between 0 and 1, where 0 means completely unrelated and 1 means semantically identical. "
            "You can ignore minor differences in grammatical or function-word usage (articles, prepositions, plurals). "
            "Output only the number — no words, no explanations."
        )

    @torch.no_grad()
    def judge(self, predicted_text: str, ground_truth_text: str) -> float:
        predicted_text = predicted_text.lower()
        ground_truth_text = ground_truth_text.lower()
        # create message template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"PREDICTED: {predicted_text}\n"
                    f"GROUND TRUTH: {ground_truth_text}\n\n"
                    "Output only the score. No text."
                ),
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            return_tensors="pt",
        ).to(self.model.device)

        # generate the output
        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )

        decoded_output = self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        decoded_output = decoded_output.replace("\n", "").strip()
        #print(f"decoded_output: {decoded_output}")
        return float(decoded_output)

if __name__ == "__main__":
    #judge = LLMJudge(model_id="Qwen/Qwen3-4B-Thinking-2507")
    judge = LLMJudge(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    answer = judge.judge("All-In in las vegas seems is a good idea.", "in las vegas All-In is a good one.")

    # judge.judge("it's a good idea.", "wow you're smart, that's a good idea.")
    print(f"Answer: {answer}")
    import pdb; pdb.set_trace()
