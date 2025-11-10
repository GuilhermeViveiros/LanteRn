"""
Dataset for supervised fine-tuning (SFT)
"""
import json
import os
from PIL import Image
from datasets import Dataset
from typing import Dict
import torch
from src.utils import center_and_crop_image


class SFTData(Dataset):
    def __init__(self, data_path: str, visual_model: torch.nn.Module):
        #super().__init__()
        # load json data
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        # visual model should be a model that takes an image and returns the patch features (the visual part of Qwen2.5VL for instance)
        self.visual_model = visual_model

    def __len__(self):
        return len(self.dataset)

    def get_visual_features(self, img: Image.Image) -> torch.Tensor:
        # get the visual features
        import pdb; pdb.set_trace()
        visual_features = self.visual_model(img)
        return visual_features

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # retrieve the image
        data = self.dataset[idx]
        # extract the image patch features
        img = Image.open(data["img_path"])
        img_bboxs = [center_and_crop_image(img, bbox) for bbox in data["bboxs"]]
        
        # lantern is divided into three components
        # 1. the vq sample (image + question)
        # 2. the optional pre-reasoning trace before the latent visual reasoning
        import pdb; pdb.set_trace()
        # get the image patch features
        img_path = os.path.join(self.img_folder_path, data["dataset"], data["image"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path {img_path} does not exist")
        img = Image.open(img_path)
        # save img
        #img.save("img.jpg")
        img_bboxes = [center_and_crop_image(img, bbox) for bbox in data["bboxs"]]    

        return Sample(
            image=img, 
            img_bboxes=img_bboxes, 
            img_path=img_path,
            question=data["question"],
            answer=data["answer"],
            dataset=data["dataset"],
            split=data["split"],
            bboxs=data["bboxs"]
        )


if __name__ == "__main__":
    data_path = "/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"

    # load the visual model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #    model_id, torch_dtype="auto", device_map="auto"
    #)
    processor = AutoProcessor.from_pretrained(model_id)
    data = SFTData(data_path, None)
    for i in range(10):
        data[i]
        break
    import pdb; pdb.set_trace()