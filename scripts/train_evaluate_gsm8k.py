import argparse
import json
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


def run_sft_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_name: str,
    prompts: list[str],
    examples: list[str],
    gradient_accumulation_steps: int,
    enable_sync_policy_weights: bool,
    vllm_base_url: str | None,
):
    assert len(prompts) == len(examples)
    responses = [extract_response(example) for example in examples]

    # print()
    # print("=" * 80)
    # print(f"prompt example: {prompts[0]}")
    # print(f"response example: {responses[0]}")
    # print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-6
    )

    print()
    print("=" * 80)
    print(f"Start training {len(prompts)} examples")
    print("=" * 80)

    total_loss, metadata = run_sft_train_step(
        model,
        tokenizer,
        optimizer,
        prompts,
        responses,
        gradient_accumulation_steps,
        max_grad_norm=1.0,
    )

    print()
    print("=" * 80)
    print(f"Complete {prompt_name} sft training, {total_loss=}")
    print("=" * 80)

    if enable_sync_policy_weights:
        print()
        print("=" * 80)
        print(f"Start sync policy weight to {vllm_base_url}")
        print("=" * 80)

        weight_sync_group = init_weight_sync(vllm_base_url, "cuda:0")
        sync_policy_weights(model, vllm_base_url, weight_sync_group)

        print()
        print("=" * 80)
        print(f"Complete sync policy weight")
        print("=" * 80)


def run_grpo_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_name: str,
    prompts: list[str],
    rollout_completions: list[list[VLLMCompletion]],
    examples: list[str],
    vllm_base_url: str,
    rollouts: int,
    gradient_accumulation_steps: int,
    enable_sync_policy_weights: bool,
):

    assert len(rollout_completions) == rollouts

    repeated_prompts = [prompt for i in range(rollouts) for prompt in prompts]
    rollout_responses = [
        completion.text
        for i in range(rollouts)
        for completion in rollout_completions[i]
    ]
    repeated_ground_truths = [
        extract_gt(example) for i in range(rollouts) for example in examples
    ]

    assert (
        len(repeated_prompts) == len(rollout_responses) == len(repeated_ground_truths)
    ), f"{len(repeated_prompts)=}, {len(rollout_responses)=}, {len(repeated_ground_truths)=}"

    n_train_examples = 6400
    n_val_examples = 1024
    num_rollout_steps = 200
    learning_rate = 1e-5
    rollout_batch_size = train_batch_size = 256
    group_size = 8
    # gradient_accumulation_steps = 32
    sampling_temperature = 1.0
    sampling_max_tokens = 512
    max_grad_norm = 1.0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    print()
    print("=" * 80)
    print(f"Start training {len(repeated_prompts)} examples")
    print("=" * 80)

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
        group_size=group_size,
        baseline="mean",
        advantage_eps=1e-6,
        advantage_normalizer="std",
        importance_reweighting_method="none",
    )

    print()
    print("=" * 80)
    print(f"Complete {prompt_name} grpo training, {total_loss=}")
    print(
        f"Rewards: reward_mean={metadata['reward_mean']}, reward_std={metadata['reward_std']}, reward_max={metadata['reward_max']}, reward_min={metadata['reward_min']}"
    )
    print(
        f"Advantages: adv_mean={metadata['adv_mean']}, adv_std={metadata['adv_std']}, adv_max={metadata['adv_max']}, adv_min={metadata['adv_min']}"
    )
    print("=" * 80)

    if enable_sync_policy_weights:
        print()
        print("=" * 80)
        print(f"Start sync policy weight to {vllm_base_url}")
        print("=" * 80)

        weight_sync_group = init_weight_sync(vllm_base_url, "cuda:0")
        sync_policy_weights(model, vllm_base_url, weight_sync_group)

        print()
        print("=" * 80)
        print(f"Complete sync policy weight")
        print("=" * 80)


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

    data = load_jsonl(GSM8K_TRAIN if args.sft_training else GSM8K_TEST)
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
        print(f"Examples: {len(prompts)}")
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

            for batch, prompts in enumerate(prompt_batches):
                if args.sft_training:
                    run_sft_train(
                        model,
                        tokenizer,
                        prompt_name,
                        prompts,
                        data_batches[batch],
                        args.gradient_accumulation_steps,
                        args.sync_policy_weight,
                        args.base_url,
                    )
                    continue

                print()
                print("=" * 80)
                print(
                    f"Generate completion for batch: {batch} with {len(prompts)} prompts and {args.rollouts} rollouts, total {len(prompt_batches)} batches"
                )
                print("=" * 80)
                rollout_completions = []
                for rollout in range(args.rollouts):
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
                        # batch_size=args.batch_size,
                    )
                    rollout_completions.append(completions)
                run_grpo_train(
                    model,
                    tokenizer,
                    prompt_name,
                    prompts,
                    rollout_completions,
                    data_batches[batch],
                    args.base_url,
                    args.rollouts,
                    args.gradient_accumulation_steps,
                    args.sync_policy_weight,
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
