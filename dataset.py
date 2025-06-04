import os
import json
from torch.utils.data import Dataset
from PIL import Image

class LLaVACoT(Dataset):
    def __init__(self, data_path: str, concise_answer: bool = False):
        self.data_path = data_path
        self.data = self.load_data()
        self.concise_answer = concise_answer
        self.single_word_or_phrase = " Answer the question using a single word or phrase."

    def load_data(self):
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        image_path = os.path.join(self.data_path.rsplit("/", 1)[0], "unzipped", data_dict["image"])
        image = Image.open(image_path)
        conversations = data_dict["conversations"]

        for conversation in conversations:
            if self.concise_answer and conversation["from"] == "human":
                if self.single_word_or_phrase in conversation["value"]:
                    conversation["value"] = conversation["value"].replace(self.single_word_or_phrase, "")
            if conversation["from"] == "gpt":
                conversations.remove(conversation)
                
        return {
            "conversations": conversations,
            "image": image
        }