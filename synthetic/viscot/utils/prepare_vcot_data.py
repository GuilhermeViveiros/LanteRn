from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset, Features, Value, Sequence, concatenate_datasets
import copy
import os
import numpy as np
from typing import List
from dataclasses import dataclass
from PIL import Image
from utils.utils import center_and_crop_image


IMG_FOLDER_PATH = "/e/project1/jureap126/gviveiros/lantern/"

if not os.path.exists(IMG_FOLDER_PATH):
    raise FileNotFoundError(f"Image folder path {IMG_FOLDER_PATH} does not exist")

def filter_bbox_assement(bboxs: List[List[float]], img_height: int, img_width: int, context_scale: float = 1.2):
    """
    Filter the bounding box based on the area and context scale.
    Args:
        bbox: Bounding box [x1, y1, x2, y2].
        area: Area of the bounding box.
        context_scale: Context scale.
    Returns:
        True if the bounding box is valid, False otherwise.
    """
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        bbox_area = w * h
        # if w or height is higher than the image, return False
        if w > img_width or h > img_height:
            return False
        if max(x1, x2) > img_width or max(y1, y2) > img_height:
            return False
        img_area = img_height * img_width
        
        if not (bbox_area * context_scale > 0.05 * img_area and bbox_area * context_scale < 0.4 * img_area):
            return False
    return True


@dataclass
class Sample:
    image: Image.Image
    img_bboxes: List[Image.Image]
    question: str
    answer: str
    dataset: str
    img_path: str
    bboxs: List[List[float]]
    split: str

    # @staticmethod
    # def to_base64(image: np.ndarray):
    #     # Convert from BGR (OpenCV) to RGB (PIL)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(image)
    #     # decode the image to base64
    #     buffer = BytesIO()
    #     image.save(buffer, format="JPEG")
    #     base64_bytes = base64.b64encode(buffer.getvalue())
    #     base64_string = base64_bytes.decode("utf-8")
    #     return f"data:image/jpeg;base64,{base64_string}"


class VCoTData(Dataset):
    def __init__(
        self,
        folder_path:str,
        split:str="train",
        shuffle:bool=False,
        verbose:bool=False,
    ):
        """
        Initialize the VCoT dataset.
        Args:
            folder_path: Path to the folder containing the dataset.
            split: Split of the dataset.
            shuffle: Whether to shuffle the dataset.

        Curating process:
        - We need to crop the images to the bounding boxes.
        - Filter every bounding box where area < 2% and > 40% of the image.
        """
        self.dataset_size = 0
        features = Features({
            "question": Value("string"),
            "answer": Value("string"),
            "image": Value("string"),
            "width": Value("int64"),
            "height": Value("int64"),
            "bboxs": Sequence(Sequence(Value("float64"))),  # list of lists of floats
            "dataset": Value("string"),
            "split": Value("string"),
        })
        dataset_list = []
        
        # get parent folder of the folder_path
        self.img_folder_path = IMG_FOLDER_PATH
        
        for file in os.listdir(folder_path):
            if file.endswith(".jsonl"):
                # if textcap is in the file name, skip it
                # TODO: still need to fix this. Textcap is not in the hf directory of viscot
                # https://huggingface.co/datasets/deepcs233/Visual-CoT/viewer/default/train
                # we need to download the original dataset and move it to the img folder of "lantern", carefull with the size of the images since we will have to crop them latter
                if "textcap" in file:
                    continue
                file_path = os.path.join(folder_path, file)
                dataset = load_dataset("json", data_files=file_path, split="train")
                if verbose:
                    print(f"Dataset from {file.split('/')[-1]} size: {len(dataset)}")
                dataset = dataset.filter(lambda x: filter_bbox_assement(x["bboxs"], x["height"], x["width"]))
                if verbose:
                    print(f"Dataset from {file.split('/')[-1]} size after filtering: {len(dataset)}")
                # from the dataset, get only the columns that are in the features
                desired_columns = list(features.keys())
                dataset = dataset.select_columns(desired_columns)
                # Cast to ensure correct types
                dataset = dataset.cast(features)
                dataset_list.append(dataset)
                self.dataset_size += len(dataset)
        
        assert len(dataset_list) > 0, f"No datasets found in the folder {folder_path}"
        # concatenate the datasets
        self.dataset = concatenate_datasets(dataset_list)

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)

        print(f"Dataset size: {len(self.dataset)}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # retrieve the image
        data = self.dataset[idx]
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

    def get_dataset_mix(self, verbose:bool=True):
        dataset_mix = []
        for sample in self.dataset:
            dataset_mix.append(sample["dataset"])
        if verbose:
            ds, ds_size = np.unique(dataset_mix, return_counts=True)
            for d, size in zip(ds, ds_size):
                print(f"Dataset: {d} - Size: {size}")
            return list(ds)
        
        sorted_mix = sorted(set(dataset_mix))
        print(f"Dataset mix: {sorted_mix}")
        return list(set(dataset_mix))

    def filter(self, filter_fn):
        """Filter the dataset and update internal state.
        
        Args:
            filter_fn: Function that takes a sample and returns True to keep it.
        
        Returns:
            self (for method chaining)
        """
        self.dataset = self.dataset.filter(filter_fn)
        self.dataset_size = len(self.dataset)
        return self

    def copy(self):
        return copy.deepcopy(self)

    def shuffle(self, seed:int=42):
        self.dataset = self.dataset.shuffle(seed=seed)
        return self

    def get_dataset_size(self):
        return self.dataset_size

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    dataset = VCoTData(
        folder_path="/home/gviveiros/.cache/huggingface/hub/datasets--deepcs233--Visual-CoT/snapshots/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata",
        split="train"
    )
    # dataset.get_dataset_mix(verbose=False)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True, collate_fn=lambda x: x)

    for batch in dataloader:
        # do a verbose print of the batch content
        print("--------------------------------")
        print("Batch content:")
        for i, sample in enumerate(batch):
            print(f"Image {i}: {sample.image.size} at img_path: {sample.img_path}")
        print("--------------------------------")
        break
        
# LANTERN -> LAteNt visual sTructurE ReasoniNg
