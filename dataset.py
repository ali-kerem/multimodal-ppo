import os
import json
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


class LLaVACoT(Dataset):
    def __init__(self, data_path: str, concise_answer: bool = True):
        self.data_path = data_path
        self.load_dataset()
        self.concise_answer = concise_answer
        self.single_word_or_phrase = " Answer the question using a single word or phrase."

    def load_dataset(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        image_path = os.path.join(self.data_path.rsplit("/", 1)[0], data_dict["image"])
        image = Image.open(image_path)

        if not self.concise_answer:
            data_dict["conversations"][0]["content"] = data_dict["conversations"][0]["content"].replace(self.single_word_or_phrase, "")

        return {
            "conversations": data_dict["conversations"][:1],
            "conversations_all": data_dict["conversations_all"],
            "image": image,
        }


class Geometry3KDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train"
    ):
        self.dataset = load_dataset(data_path)[split]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        conv = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["images"][0]
                    },
                    {
                        "type": "text",
                        "text": sample["problem"]
                    }
                ]
            }
        ]

        return {
            "conversations": conv,
            "answer": sample["answer"]
        }
