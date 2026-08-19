from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    assert len(prompt_strs) == len(output_strs)

    tokenized_data = [
        (tokenizer.encode(prompt), tokenizer.encode(output))
        for prompt, output in zip(prompt_strs, output_strs)
    ]

    max_prompt_and_output_len = (
        max([len(data[0]) + len(data[1]) for data in tokenized_data]) - 1
    )

    def pad(tensor):
        if len(tensor) < max_prompt_and_output_len:
            tensor = torch.concat(
                [
                    tensor,
                    torch.zeros(
                        max_prompt_and_output_len - len(tensor), dtype=torch.long
                    ),
                ]
            )
        if len(tensor) > max_prompt_and_output_len:
            tensor = tensor[:max_prompt_and_output_len]

        return tensor

    input_ids = []
    labels = []
    masks = []
    for data in tokenized_data:
        input_id = torch.tensor(data[0] + data[1], dtype=torch.long)
        label = input_id[1:]
        mask = torch.concat([torch.zeros(len(data[0]) - 1), torch.ones(len(data[1]))])

        input_ids.append(pad(input_id))
        labels.append(pad(label))
        masks.append(pad(mask))

    return {
        "input_ids": torch.stack(input_ids).to(device="cuda"),
        "labels": torch.stack(labels).to(device="cuda"),
        "response_mask": torch.stack(masks).to(device="cuda"),
    }


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    ret = model(input_ids=input_ids, labels=labels)
    all_log_probs = torch.log_softmax(ret.logits, dim=-1)
    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    outputs: dict[str, Tensor] = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        probs = all_log_probs.exp()
        token_entropy = -(probs * all_log_probs).sum(dim=-1)
        outputs["token_entropy"] = token_entropy

    return outputs


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    prompt_name: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    assert len(rollout_responses) == len(repeated_ground_truths)

    raw_rewards = []
    format_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_output = reward_fn(prompt_name, response, gt)
        raw_rewards.append(reward_output["reward"])
        format_rewards.append(reward_output["format_reward"])

    return (
        torch.tensor(raw_rewards),
        {
            "reward": raw_rewards,
            "meam_total_reward": sum(raw_rewards) / len(raw_rewards),
            "mean_format_reward": sum(format_rewards) / len(format_rewards),
        },
    )


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    grouped_rewards = raw_rewards.reshape(-1, group_size)

    if advantage_normalizer == "std":
        reward_norm = grouped_rewards.std(dim=1, keepdim=True)
    elif advantage_normalizer == "mean":
        reward_norm = grouped_rewards.mean(dim=1, keepdim=True)
    else:
        reward_norm = torch.ones_like(grouped_rewards[:, :1])

    if baseline == "mean":
        reward_baseline = grouped_rewards.mean(dim=1, keepdim=True)
    else:
        reward_baseline = torch.zeros_like(grouped_rewards[:, :1])

    advantages = (grouped_rewards - reward_baseline) / (reward_norm + advantage_eps)

    return (
        advantages.flatten().to(device="cuda"),
        {
            "adv_mean": grouped_rewards.mean(),
            "adv_std": grouped_rewards.std(),
            "adv_max": grouped_rewards.max(),
            "adv_min": grouped_rewards.min(),
        },
    )


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if importance_reweighting_method == "none":
        loss = -raw_rewards_or_advantages.reshape(-1, 1) * policy_log_probs
    else:
        raise NotImplementedError

    return (loss, {})


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "constant":
        assert normalization_constant is not None
        loss /= normalization_constant
    elif loss_normalization == "sequence":
        loss = loss.sum(dim=1) / mask.sum(dim=1)

    return loss.mean()


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    prompt_name: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """

    tokenizer_outputs = run_tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )
    input_ids = tokenizer_outputs["input_ids"]
    labels = tokenizer_outputs["labels"]
    response_mask = tokenizer_outputs["response_mask"]

    rewards, reward_metadata = run_compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths, prompt_name
    )

    advantages, advantages_metadata = run_compute_group_normalized_rewards(
        rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )

    batch_size = input_ids.size(0)
    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    metadata = {}

    total_loss = 0.0
    for mb in range(gradient_accumulation_steps):
        mb_start = mb * microbatch_size
        mb_end = mb_start + microbatch_size

        mb_input_ids = input_ids[mb_start:mb_end]
        mb_labels = labels[mb_start:mb_end]

        model_outputs = run_get_response_log_probs(
            model, mb_input_ids, mb_labels, False
        )
        log_probs = model_outputs["log_probs"]

        mb_old_log_probs = (
            old_log_probs[mb_start:mb_end] if old_log_probs is not None else None
        )

        pg_loss, pg_loss_metadata = run_compute_policy_gradient_loss(
            advantages[mb_start:mb_end],
            log_probs,
            importance_reweighting_method,
            mb_old_log_probs,
            cliprange,
            response_mask[mb_start:mb_end],
        )

        loss = run_aggregate_loss_across_microbatch(
            pg_loss,
            response_mask[mb_start:mb_end],
            loss_normalization,
            normalization_constant,
        )

        loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.detach()

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata.update(reward_metadata)
    metadata.update(advantages_metadata)
    metadata["grad_norm"] = grad_norm

    return (total_loss, metadata)


def run_sft_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompts: list[str],
    responses: list[str],
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
):
    tokenizer_outputs = run_tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = tokenizer_outputs["input_ids"]
    labels = tokenizer_outputs["labels"]

    batch_size = input_ids.size(0)
    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    metadata = {}

    total_loss = 0.0
    for mb in range(gradient_accumulation_steps):
        mb_start = mb * microbatch_size
        mb_end = mb_start + microbatch_size

        mb_input_ids = input_ids[mb_start:mb_end]
        mb_labels = labels[mb_start:mb_end]

        model_outputs = run_get_response_log_probs(
            model, mb_input_ids, mb_labels, False
        )
        log_probs = model_outputs["log_probs"]

        loss = -log_probs

        loss = loss.mean() / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.detach()

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata["grad_norm"] = grad_norm

    return (total_loss, metadata)


"""
The below adapters are used in the optional
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
