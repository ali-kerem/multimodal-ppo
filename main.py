import os
import warnings
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from dataset import LLaVACoT
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from MultimodalPPOTrainer import MultimodalPPOTrainer
from llava.model.builder import load_pretrained_model
from data_collator import LLaVACoTDataCollator

warnings.filterwarnings("ignore")


def main():
    # Here we will do the following:
    # 1. Load dataset
    # 2. Load Actor model
    # 3. Load Critic model
    # 4. Load Value model
    # 5. Initialize PPO Config
    # 6. Initialize PPO trainer

    # 1. Load dataset
    home_dir = os.path.expanduser("~")
    data_path = os.path.join(home_dir, "/datasets/LLaVA-CoT-100k/train.json")
    dataset = LLaVACoT(data_path=data_path)

    # 2. Load Actor model
    actor_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    actor_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    actor_model_name, torch_dtype="auto", device_map="auto")
    actor_processor = AutoProcessor.from_pretrained(actor_model_name)
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    actor_model_name, torch_dtype="auto", device_map="auto")

    # 3. Load Reward model
    reward_model_path = "lmms-lab/llava-critic-7b"
    reward_model_name = "llava_qwen"
    reward_device = "cuda"
    reward_device_map = "auto"
    reward_tokenizer, reward_model, reward_image_processor, reward_max_length = load_pretrained_model(reward_model_path, None, reward_model_name, device_map=reward_device_map)

    # 4. Load Value model
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(actor_model_name, torch_dtype="auto", device_map="auto")

    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=1e-5)

    data_collator = LLaVACoTDataCollator(tokenizer=actor_processor)

    # 4. Initialize PPO Config
    ppo_config = {"mini_batch_size": 1, "batch_size": 1, "report_to": "none"}
    config = PPOConfig(**ppo_config)
    ppo_trainer = MultimodalPPOTrainer(args=config,
                             model=actor_model,
                             ref_model=ref_model,
                             processing_class=reward_tokenizer,
                             train_dataset=dataset,
                             reward_model=reward_model,
                             value_model=value_model,
                             optimizers=(optimizer, None),
                             data_collator=data_collator)

    ppo_trainer.train()


if __name__ == "__main__":
    main()

