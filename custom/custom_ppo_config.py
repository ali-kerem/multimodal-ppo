import os
from dataclasses import dataclass
from trl import PPOConfig
from accelerate import PartialState

@dataclass
class CustomPPOConfig(PPOConfig):
    def __init__(self, accelerator, **kwargs):
        self.accelerator = accelerator
        super().__init__(**kwargs)

    def __post_init__(self):
        # The TRL PPOConfig inherits from TrainingArguments, which has a __post_init__
        # that conflicts with `accelerate launch`. We override it here to prevent
        # that from running.
        
        # From OnPolicyConfig
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        
        # Minimal bits from TrainingArguments
        if self.output_dir is None:
            self.output_dir = "trainer_output"
        
        if self.logging_dir is None and self.output_dir is not None:
            from transformers.training_args import default_logdir
            self.logging_dir = os.path.join(self.output_dir, default_logdir())

        # Manually set the distributed state that Trainer needs
        self.distributed_state = self.accelerator.state # Should keep like this?
