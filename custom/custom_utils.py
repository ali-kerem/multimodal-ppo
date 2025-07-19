import torch
from typing import List
from transformers import GenerationConfig
from transformers.utils import ModelOutput

def get_reward(response_text: List[str], answer_text: List[str]) -> List[float]:
    rewards = []
    for response, answer in zip(response_text, answer_text):
        idx = response.find("Answer:")
        if idx == -1 or "</think>" not in response:
            rewards.append(0.0)
            continue
        else:
            response = response[idx + len("Answer:") :]
        if answer in response:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)

def get_value(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the value logits for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        tuple:
            - `value_logits` (`torch.Tensor`):
                The logits for the value model.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    value_logits = model.score(output.hidden_states[-1])
    return value_logits

def forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    pad_token_id: int,
) -> ModelOutput:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        inputs (`dict[str, torch.Tensor]`):
            The dictionary containing the input tensors (input_ids, pixel_values, image_grid_thw).
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `ModelOutput`:
            The output of the model, including hidden states.
    """
    attention_mask = inputs["input_ids"] != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    inputs["input_ids"] = torch.masked_fill(inputs["input_ids"], ~attention_mask, 0)
    return model(
        **inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def generate(
    lm_backbone: torch.nn.Module, queries: dict[str, torch.Tensor], generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    output = lm_backbone.generate(
        **queries,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return output.sequences, logits
