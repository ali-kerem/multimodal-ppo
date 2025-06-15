import os
import json
from torch.utils.data import Dataset
from PIL import Image
from processor import ActorProcessor


class LLaVACoT(Dataset):
    def __init__(self, data_path: str, concise_answer: bool = True, model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.data_path = data_path
        self.load_dataset()
        self.concise_answer = concise_answer
        self.single_word_or_phrase = " Answer the question using a single word or phrase."
        self.preprocessor = ActorProcessor(model_name_or_path)

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

        input_ids = self.preprocessor.preprocess_data(image, data_dict["conversations"][:1], add_generation_prompt=True)
        input_ids_all = self.preprocessor.preprocess_data(image, data_dict["conversations"], add_generation_prompt=False)
                
        return {
            "input_ids": input_ids,
            "input_ids_all": input_ids_all,
            "image": image,
        }
