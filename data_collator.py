from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
from qwen_vl_utils import process_vision_info

@dataclass
class LLaVACoTDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [
            self.processor.apply_chat_template(
                sample["conversations"],
                tokenize=False,
                add_generation_prompt=True
            )
            for sample in batch
        ]
        texts_all = [
            self.processor.apply_chat_template(
                sample["conversations_all"],
                tokenize=False,
                add_generation_prompt=False
            )
            for sample in batch
        ]

        images = [process_vision_info(sample["conversations"])[0] for sample in batch]

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )

        input_ids_all = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )
        
        return {
            "inputs": {
                "input_ids":      model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "pixel_values":   model_inputs["pixel_values"],
                "image_grid_thw": model_inputs["image_grid_thw"],
            },
            "inputs_all": {
                "input_ids":      input_ids_all["input_ids"],
                "attention_mask": input_ids_all["attention_mask"],
                "pixel_values":   input_ids_all["pixel_values"],
                "image_grid_thw": input_ids_all["image_grid_thw"],
            }
        }
