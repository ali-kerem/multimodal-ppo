import os
import json
from torch.utils.data import Dataset


class LLaVACoT(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()
        self.start_tag = "<CONCLUSION>"
        self.end_tag = "</CONCLUSION>"

    def load_data(self):
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                data.append(json.loads(line)) # Use json.loads() for each line
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        image_path = os.path.join(self.data_path, data_dict["image"])
        conversations = data_dict["conversations"]
        for conversation in conversations:
            if conversation["from"] == "gpt":
                no_reasoning_answer = conversation["value"]
                start_index = no_reasoning_answer.find(self.start_tag) + len(self.start_tag)
                end_index = no_reasoning_answer.find(self.end_tag)
                conclusion_text = no_reasoning_answer[start_index:end_index].strip()
                conversation["value"] = conclusion_text
                
        return conversations, image_path
