from transformers import AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info

class ActorProcessor:
    def __init__(self, model_name_or_path: str):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def preprocess_data(self, image: Image.Image = None, conversations: list[dict] = None, add_generation_prompt: bool = True):
        text = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if image is not None:
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",)
        else:
            inputs = self.processor(
                text=text,
                padding=False)
        return inputs['input_ids']
    