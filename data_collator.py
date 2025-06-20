from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase

@dataclass
class MultimodalDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [f["image"] for f in features]
        
        input_ids_features = [{"input_ids": f["input_ids"]} for f in features]
        padded_input_ids = self.tokenizer.pad(
            input_ids_features,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        input_ids_all_features = [{"input_ids": f["input_ids_all"]} for f in features]
        padded_input_ids_all = self.tokenizer.pad(
            input_ids_all_features,
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )
        
        return {
            "input_ids": padded_input_ids["input_ids"],
            "attention_mask": padded_input_ids["attention_mask"],
            "input_ids_all": padded_input_ids_all["input_ids"],
            "images": images,
        }
