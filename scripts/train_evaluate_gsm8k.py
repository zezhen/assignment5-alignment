import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from cs336_alignment.adapters import run_grpo_train_step, run_sft_train_step
from cs336_alignment.checkpoint import get_model_and_tokenizer, save_model_and_tokenizer
from cs336_alignment.drgrpo_grader import extract_answer, grade
from cs336_alignment.vllm_utils import (
    generate_completions,
    init_weight_sync,
    sync_policy_weights,
    VLLMCompletion,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase


ROOT = Path(__file__).resolve().parent.parent

PROMPTS = {
    "question_only": ROOT / "cs336_alignment/prompts/question_only.prompt",
    "r1_zero": ROOT / "cs336_alignment/prompts/r1_zero.prompt",
    "r1_zero_three_shot": (
        ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
    ),
}

GSM8K_TEST = ROOT / "data/gsm8k/test.jsonl"
GSM8K_TRAIN = ROOT / "data/gsm8k/train.jsonl"


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_prompt(path: Path) -> str:
    return path.read_text()


def extract_gt(example: dict) -> str:
    """
    GSM8K answers conventionally look like:

        ... reasoning ...
        #### 42

    Return the final answer.
    """
    answer = example["answer"]
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip()


def extract_response(example: dict) -> str:
    answer = example["answer"]

    assert "####" in answer

    splits = answer.split("####")

    answer = splits[0]
    final_answer = splits[-1].strip().replace(",", "").strip()

    return f"<think>{answer}</think><answer>{final_answer}</answer>"


def parse_question_only(output: str):
    """
    question_only.prompt explicitly requests \\boxed{}.
    """
    try:
        return extract_answer(output)
    except Exception:
        return None


def parse_r1(output: str):
    """
    Expected completion after prompt ends in:
        Assistant: <think>

    So the generated text should contain:
        reasoning </think> <answer> ANSWER </answer>
    """
    if "</think>" not in output:
        return None

    if "<answer>" not in output or "</answer>" not in output:
        return None

    answer = output.split("<answer>", 1)[1].split("</answer>", 1)[0]
    answer = answer.strip()

    return answer or None


def format_reward(prompt_name: str, output: str) -> int:
    if prompt_name == "question_only":
        return int(parse_question_only(output) is not None)

    # Since the prompt itself already ends with "<think>",
    # output starts *inside* the think section.
    return int("</think>" in output and "<answer>" in output and "</answer>" in output)


def parsed_answer(prompt_name: str, output: str):
    if prompt_name == "question_only":
        return parse_question_only(output)

    return parse_r1(output)


def score(prompt_name: str, output: str, gt: str):
    fmt = format_reward(prompt_name, output)

    pred = parsed_answer(prompt_name, output)

    if pred is None:
        answer = 0
    else:
        try:
            answer = int(grade(pred, gt))
        except Exception:
            answer = 0

    return {
        "format_reward": fmt,
        "correctness_reward": answer,
        "parsed_answer": pred,
    }


def reward(prompt_name: str, output: str, gt: str) -> dict[str, float]:
    rewards = score(
        prompt_name=prompt_name,
        output=output,
        gt=gt,
    )

    return {
        "reward": rewards["correctness_reward"] * 1.0,
        "format_reward": rewards["format_reward"] * 1.0,
    }


def format_grad_norms(grad_norms: list, max_grad_norm: float | None) -> str:
    """Pre-clip gradient norms, flagged with * where the clip rescaled the step."""
    if not grad_norms or grad_norms[0] is None:
        return "gnorm=off"

    values = [norm.item() for norm in grad_norms]
    clipped = [
        f"{value:.3f}{'*' if max_grad_norm is not None and value > max_grad_norm else ''}"
        for value in values
    ]

    return f"gnorm={sum(values) / len(values):.3f} [{' '.join(clipped)}]"


def category(fmt: int, correct: int):
    if fmt == 1 and correct == 1:
        return 1
    elif fmt == 1 and correct == 0:
        return 2
    elif fmt == 0 and correct == 0:
        return 3

    # Shouldn't normally happen because a malformed answer can't
    # receive correctness reward under this parser.
    return 4


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k for a single problem: 1 - C(n-c, k) / C(n, k).

    Chen et al. 2021 eq. 1, in the product form that cannot overflow. Estimating
    pass@k from n >> k samples has far lower variance than asking whether the
    first k of n happened to contain a correct one.
    """
    if k > num_samples:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {num_samples}")
    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - math.prod(
        1.0 - k / n for n in range(num_samples - num_correct + 1, num_samples + 1)
    )


def pass_at_k_report(
    correct_per_group: list[int],
    group_size: int,
    ks: list[int],
) -> dict[int, float]:
    """Average unbiased pass@k over problems, for each requested k."""
    return {
        k: sum(pass_at_k(group_size, c, k) for c in correct_per_group)
        / len(correct_per_group)
        for k in ks
        if k <= group_size
    }


def correct_per_group(answer_rewards: list[float], group_size: int) -> list[int]:
    """Number of correct rollouts in each group, in group-major order."""
    return [
        sum(1 for r in answer_rewards[start : start + group_size] if r > 0)
        for start in range(0, len(answer_rewards), group_size)
    ]


def format_pass_at_k(report: dict[int, float]) -> str:
    return " ".join(f"p@{k}={value:.3f}" for k, value in sorted(report.items()))


def evaluate_pass_at_k(
    prompt_name: str,
    template: str,
    examples: list[dict],
    vllm_base_url: str,
    model_id: str,
    samples: int,
    max_tokens: int,
    temperature: float,
    ks: list[int],
) -> dict[int, float]:
    """pass@k on a held-out set, sampled fresh from whatever vLLM is serving.

    Sampling temperature has to be > 0: at 0 every one of the k samples is the
    same string and pass@k collapses onto pass@1 for every k.
    """
    completions = generate_completions(
        vllm_base_url=vllm_base_url,
        model_id=model_id,
        prompts=[template.format(question=example["question"]) for example in examples],
        sampling_params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": samples,
            "seed": None,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
    )
    assert len(completions) == len(examples) * samples

    counts = []
    for index, example in enumerate(examples):
        gt = extract_gt(example)
        group = completions[index * samples : (index + 1) * samples]
        counts.append(
            sum(score(prompt_name, completion.text, gt)["correctness_reward"] for completion in group)
        )

    return pass_at_k_report(counts, samples, ks)


def run_sft_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompt_name: str,
    prompts: list[str],
    examples: list[str],
    gradient_accumulation_steps: int,
    enable_sync_policy_weights: bool,
    vllm_base_url: str | None,
    batch_index: int,
    weight_sync_group,
    epochs: int = 1,
):
    assert len(prompts) == len(examples)
    responses = [extract_response(example) for example in examples]

    total_loss, metadata = run_sft_train_step(
        model,
        tokenizer,
        optimizer,
        prompts,
        responses,
        gradient_accumulation_steps,
        max_grad_norm=1.0,
        epochs=epochs,
    )

    losses = " ".join(f"{loss.item():.4f}" for loss in metadata["epoch_losses"])
    grads = format_grad_norms(metadata["epoch_grad_norms"], 1.0)
    print(
        f"sft training batch {batch_index}, loss={total_loss.item():.4f} [{losses}], {grads}"
    )

    if enable_sync_policy_weights:
        sync_policy_weights(model, vllm_base_url, weight_sync_group)

def build_grpo_batch(
    completions: list[VLLMCompletion],
    device: str = "cuda",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Tokenized batch plus the sampler's own log-probs, built from vLLM's ids.

    Re-encoding the decoded text disagrees with the ids vLLM actually sampled on
    roughly a tenth of rollouts -- sometimes a different length, sometimes the
    same length with different ids -- which silently misaligns the two log-prob
    tensors against each other. Taking the ids straight from the engine makes
    every row exactly len(prompt) + len(generated) - 1 by construction.
    """
    rows = []
    for completion in completions:
        prompt_ids = list(completion.prompt_token_ids)
        gen_ids = list(completion.token_ids)
        ids = prompt_ids + gen_ids

        # vLLM reports no log-prob for the very first prompt token, so
        # prompt_logprobs already arrives one shorter than prompt_ids.
        assert len(completion.prompt_logprobs) == len(prompt_ids) - 1
        assert len(completion.token_logprobs) == len(gen_ids)

        rows.append(
            (
                ids[:-1],
                ids[1:],
                [0.0] * (len(prompt_ids) - 1) + [1.0] * len(gen_ids),
                list(completion.prompt_logprobs) + list(completion.token_logprobs),
            )
        )

    width = max(len(row[0]) for row in rows)

    def stack(index: int, fill, dtype) -> torch.Tensor:
        return torch.tensor(
            [row[index] + [fill] * (width - len(row[index])) for row in rows],
            dtype=dtype,
            device=device,
        )

    batch = {
        "input_ids": stack(0, 0, torch.long),
        "labels": stack(1, 0, torch.long),
        "response_mask": stack(2, 0.0, torch.float),
    }

    return batch, stack(3, 0.0, torch.float)

def run_grpo_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompt_name: str,
    prompts: list[str],
    completions: list[VLLMCompletion],
    examples: list[str],
    vllm_base_url: str,
    rollouts: int,
    gradient_accumulation_steps: int,
    enable_sync_policy_weights: bool,
    batch_index: int,
    weight_sync_group,
    epochs: int = 1,
    tis_clip: float | None = 2.0,
    pass_ks: list[int] | None = None,
    max_grad_norm = 1.0,
):

    repeated_prompts = [prompt for prompt in prompts for i in range(rollouts)]
    rollout_responses = [
        completion.text
        for completion in completions
    ]
    repeated_ground_truths = [
        extract_gt(example) for example in examples for i in range(rollouts)
    ]

    assert (
        len(repeated_prompts) == len(rollout_responses) == len(repeated_ground_truths)
    ), f"{len(repeated_prompts)=}, {len(rollout_responses)=}, {len(repeated_ground_truths)=}"

    batch, vllm_log_probs = build_grpo_batch(completions)

    total_loss, metadata = run_grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        prompt_name=prompt_name,
        reward_fn=reward,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=rollouts,
        baseline="mean",
        advantage_eps=1e-6,
        advantage_normalizer="std",
        importance_reweighting_method="grpo",
        cliprange=0.2,
        epochs=epochs,
        batch=batch,
        vllm_log_probs=vllm_log_probs if tis_clip else None,
        tis_clip=tis_clip,
    )

    losses = " ".join(f"{loss.item():.4f}" for loss in metadata["epoch_losses"])
    tis = f", tis={metadata['tis_mean']:.4f}/{metadata['tis_clip_frac']:.4f}" if "tis_mean" in metadata else ""

    # Free: the group_size rollouts we just trained on are exactly the samples
    # pass@k needs. Only covers this batch's prompts, so it is as noisy as the
    # reward -- the held-out eval is the curve to read.
    passk = pass_at_k_report(
        correct_per_group(metadata["answer_rewards"], rollouts),
        rollouts,
        pass_ks or [1],
    )

    grads = format_grad_norms(metadata["epoch_grad_norms"], max_grad_norm)

    print(
        f"grpo training batch {batch_index}, loss={total_loss.item():.4f} [{losses}], "
        f"reward={metadata['reward_mean']:.4f}{tis}, {format_pass_at_k(passk)}, {grads}"
    )

    if enable_sync_policy_weights:
        sync_policy_weights(model, vllm_base_url, weight_sync_group)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-0425-1B",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--prompt",
        choices=list(PROMPTS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sft-training", type=bool, default=False)
    parser.add_argument("--grpo-training", type=bool, default=False)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--epoches", type=int, default=1)
    parser.add_argument("--tis-clip", type=float, default=2.0)
    parser.add_argument(
        "--pass-at-k",
        default="1,4,8",
        help="comma-separated k values; those above the sample count are dropped",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="held-out pass@k every N batches; 0 disables",
    )
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=8)
    # Must stay > 0: at temperature 0 all k samples are identical and every
    # pass@k collapses onto pass@1.
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sync-policy-weight", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument(
        "--ref-model",
        default="/home/zezhen/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/gsm8k_eval",
    )

    args = parser.parse_args()

    data = load_jsonl(GSM8K_TRAIN if args.sft_training or args.grpo_training else GSM8K_TEST)
    if args.limit is not None:
        data = data[: args.limit]

    prompt_names = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_name in prompt_names:
        template = load_prompt(PROMPTS[prompt_name])

        prompts = [template.format(question=example["question"]) for example in data]

        print()
        print("=" * 80)
        print(f"Evaluating: {prompt_name}")
        print(f"Examples: {len(prompts)}, batches: {args.batch_size}")
        print("=" * 80)

        if args.grpo_training or args.sft_training:
            prompt_batches = [
                prompts[start : start + args.batch_size]
                for start in range(0, len(prompts), args.batch_size)
            ]
            data_batches = [
                data[start : start + args.batch_size]
                for start in range(0, len(data), args.batch_size)
            ]

            print()
            print("=" * 80)
            print(f"Start load model")
            print("=" * 80)

            model, tokenizer = get_model_and_tokenizer(args.ref_model, "cuda")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5 if args.sft_training else 1e-5, betas=(0.9, 0.95), weight_decay=1e-6
            )

            print(
                f"Model training {len(prompt_batches)} batches with total {len(prompts)} prompts and {args.rollouts} rollouts"
            )

            weight_sync_group = None
            if args.sync_policy_weight:
                weight_sync_group = init_weight_sync(args.base_url, "cuda:0")

            pass_ks = [int(k) for k in args.pass_at_k.split(",") if k.strip()]
            eval_examples = load_jsonl(GSM8K_TEST)[: args.eval_examples]

            for batch_index, batch_prompts in enumerate(prompt_batches):
                # Goes through vLLM, so it scores whatever the server is serving:
                # without --sync-policy-weight this measures the frozen sampler,
                # not the model being trained.
                if args.eval_every and batch_index % args.eval_every == 0:
                    report = evaluate_pass_at_k(
                        prompt_name=prompt_name,
                        template=template,
                        examples=eval_examples,
                        vllm_base_url=args.base_url,
                        model_id=args.model,
                        samples=args.eval_samples,
                        max_tokens=args.max_tokens,
                        temperature=args.eval_temperature,
                        ks=pass_ks,
                    )
                    print(
                        f"eval batch {batch_index} on {len(eval_examples)} held-out "
                        f"examples x{args.eval_samples}: {format_pass_at_k(report)}"
                    )

                if args.sft_training:
                    run_sft_train(
                        model,
                        tokenizer,
                        optimizer,
                        prompt_name,
                        batch_prompts,
                        data_batches[batch_index],
                        args.gradient_accumulation_steps,
                        args.sync_policy_weight,
                        args.base_url,
                        batch_index,
                        weight_sync_group,
                        epochs=args.epoches,
                    )
                    continue

                if args.grpo_training:
                    completions: list[VLLMCompletion] = generate_completions(
                        vllm_base_url=args.base_url,
                        model_id=args.model,
                        prompts=batch_prompts,
                        sampling_params={
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "n": args.rollouts,
                            "seed": None,
                            "logprobs": 0,
                            "prompt_logprobs": 0,
                            "stop": ["</answer>"],
                            "include_stop_str_in_output": True,
                        },
                    )

                    run_grpo_train(
                        model,
                        tokenizer,
                        optimizer,
                        prompt_name,
                        batch_prompts,
                        completions,
                        data_batches[batch_index],
                        args.base_url,
                        args.rollouts,
                        args.gradient_accumulation_steps,
                        args.sync_policy_weight,
                        batch_index,
                        weight_sync_group,
                        epochs=args.epoches,
                        tis_clip=args.tis_clip,
                        pass_ks=pass_ks,
                    )

            if args.save:
                print()
                print("=" * 80)
                print(f"save latest model and tokenizer to {args.output_dir}")
                print("=" * 80)
                save_model_and_tokenizer(model, tokenizer, args.output_dir)

            continue

        completions: list[VLLMCompletion] = generate_completions(
            vllm_base_url=args.base_url,
            model_id=args.model,
            prompts=prompts,
            sampling_params={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "n": 1,
                "seed": args.seed,
            },
            batch_size=args.batch_size,
        )
        assert len(completions) == len(data)

        results = []
        counts = Counter()

        for i, (example, completion) in enumerate(zip(data, completions)):
            gt = extract_gt(example)
            output = completion.text

            rewards = score(
                prompt_name=prompt_name,
                output=output,
                gt=gt,
            )

            cat = category(
                rewards["format_reward"],
                rewards["correctness_reward"],
            )

            counts[cat] += 1

            results.append(
                {
                    "index": i,
                    "question": example["question"],
                    "ground_truth": gt,
                    "output": output,
                    "parsed_answer": rewards["parsed_answer"],
                    "format_reward": rewards["format_reward"],
                    "correctness_reward": rewards["correctness_reward"],
                    "category": cat,
                    "finish_reason": completion.finish_reason,
                }
            )

        print(f"category 1 (format=1, correct=1): {counts[1]}")
        print(f"category 2 (format=1, correct=0): {counts[2]}")
        print(f"category 3 (format=0, correct=0): {counts[3]}")

        path = args.output_dir / f"{prompt_name}.jsonl"

        with open(path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
