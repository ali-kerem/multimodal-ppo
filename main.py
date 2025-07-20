import types

import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import AutoModelForCausalLMWithValueHead

from MultimodalPPOTrainer import MultimodalPPOTrainer
from dataset import LLaVACoT, Geometry3KDataset
from data_collator import LLaVACoTDataCollator, Geometry3KDataCollator
from custom.custom_ppo_config import CustomPPOConfig


def main():
    # Here we will do the following:
    # 1. Load dataset
    # 2. Load Actor model
    # 3. Load Value model
    # 4. Initialize PPO Config
    # 5. Initialize PPO trainer

    # 1. Load dataset
    data_path = "hiyouga/geometry3k"
    train_dataset = Geometry3KDataset(data_path=data_path, split="train")
    eval_dataset = Geometry3KDataset(data_path=data_path, split="validation")

    # 2. Load Actor model
    actor_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    actor_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    actor_model_name, torch_dtype="auto", device_map="auto")
    actor_processor = AutoProcessor.from_pretrained(actor_model_name)
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    actor_model_name, torch_dtype="auto", device_map="auto")

    # 4. Load Value model
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(actor_model, torch_dtype="auto", device_map="auto")
    value_model.base_model_prefix = actor_model.base_model_prefix
    setattr(value_model, value_model.base_model_prefix, value_model.pretrained_model)

    def score(self, hidden_states):
        return self.v_head(hidden_states)
    
    value_model.score = types.MethodType(score, value_model)

    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=1e-5)

    data_collator = Geometry3KDataCollator(processor=actor_processor)

    # 4. Initialize PPO Config
    ppo_config = {"mini_batch_size": 1, "batch_size": 1, "report_to": "none"}
    accelerator = Accelerator(gradient_accumulation_steps=ppo_config["gradient_accumulation_steps"])
    config = CustomPPOConfig(accelerator=accelerator, **ppo_config)

    ppo_trainer = MultimodalPPOTrainer(args=config,
                             model=actor_model,
                             ref_model=ref_model,
                             processing_class=actor_processor,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             reward_model=None,
                             value_model=value_model,
                             optimizers=(optimizer, None),
                             data_collator=data_collator,
                             accelerator=accelerator)

    ppo_trainer.train()


if __name__ == "__main__":
    main()

