import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from dataset import LLaVACoT, Geometry3KDataset
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from MultimodalPPOTrainer import MultimodalPPOTrainer
from data_collator import LLaVACoTDataCollator, Geometry3KDataCollator


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

    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=1e-5)

    data_collator = Geometry3KDataCollator(tokenizer=actor_processor)

    # 4. Initialize PPO Config
    ppo_config = {"mini_batch_size": 1, "batch_size": 1, "report_to": "none"}
    config = PPOConfig(**ppo_config)
    ppo_trainer = MultimodalPPOTrainer(args=config,
                             model=actor_model,
                             ref_model=ref_model,
                             processing_class=actor_processor,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             reward_model=None,
                             value_model=value_model,
                             optimizers=(optimizer, None),
                             data_collator=data_collator)

    ppo_trainer.train()


if __name__ == "__main__":
    main()

