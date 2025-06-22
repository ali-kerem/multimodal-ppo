import os
import json
from torch.utils.data import Dataset
from PIL import Image


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
