"""Fine-tune an embedding model with PEFT LoRA and contrastive losses.

This script supports:

- `triplet`: expects `anchor`, `positive`, and `negative`
- `mnrl`: expects pair data such as `text_a`, `text_b`, `label`, then keeps
  only positive pairs and trains with in-batch negatives (all embeddings
  computed in a single forward pass)
- `cached_mnrl`: same as `mnrl` but computes embeddings in mini-chunks to
  reduce peak GPU memory usage

Example:
    CUDA_VISIBLE_DEVICES=1 python /tmp2/richard/esg/train_unsloth_triplet.py \
        --train-path /tmp2/richard/esg/contrastive_data/triplets.jsonl \
        --output-dir /tmp2/richard/esg/triplet_lora_qwen \
        --model-name Qwen/Qwen3-Embedding-4B \
        --trust-remote-code
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import typing

import datasets  # type: ignore[import-untyped]
import peft
import sentence_transformers
import sentence_transformers.losses
import sentence_transformers.training_args
import torch
import transformers


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Holds LoRA fine-tuning settings."""

    train_path: pathlib.Path
    output_dir: pathlib.Path
    model_name: str
    loss_type: str
    max_seq_length: int
    trust_remote_code: bool
    load_in_4bit: bool
    load_in_8bit: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    eval_ratio: float
    random_seed: int
    triplet_margin: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an embedding model with PEFT LoRA and contrastive loss."
        )
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to the JSONL training dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints and final adapters.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-4B",
        help="Base embedding model name.",
    )
    parser.add_argument(
        "--loss-type",
        default="triplet",
        choices=["triplet", "mnrl", "cached_mnrl"],
        help="Training loss type.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the base model loader.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with 4-bit quantization.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the base model with 8-bit quantization.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--target-module",
        action="append",
        default=[],
        help=(
            "LoRA target module. Repeat to override the default attention/"
            "MLP modules."
        ),
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio expressed as a fraction of total steps.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=5.0,
        help="Train epochs when max_steps is 0.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional max training steps. 0 disables the override.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save frequency in steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in the range [0, 1).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=1.0,
        help="Triplet loss margin.",
    )
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Build a frozen runtime config."""
    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be in the range [0, 1).")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load-in-4bit/--load-in-8bit.")

    target_modules = args.target_module or DEFAULT_TARGET_MODULES

    return TrainConfig(
        train_path=pathlib.Path(args.train_path),
        output_dir=pathlib.Path(args.output_dir),
        model_name=args.model_name,
        loss_type=args.loss_type,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_ratio=args.eval_ratio,
        random_seed=args.random_seed,
        triplet_margin=args.triplet_margin,
    )


def load_dataset_for_loss(config: TrainConfig) -> datasets.DatasetDict:
    """Load training data in the format required by the configured loss."""
    dataset_dict = datasets.load_dataset(
        "json",
        data_files={"train": str(config.train_path)},
    )
    train_dataset = dataset_dict["train"]
    if config.loss_type == "triplet":
        required_columns = ["anchor", "positive", "negative"]
        keep_columns = required_columns
    elif config.loss_type in ("mnrl", "cached_mnrl"):
        # Accept either anchor/positive directly, or text_a/text_b/label (filter label==1)
        if "anchor" in train_dataset.column_names and "positive" in train_dataset.column_names:
            required_columns = ["anchor", "positive"]
            keep_columns = required_columns
        else:
            required_columns = ["text_a", "text_b", "label"]
            keep_columns = required_columns

    for column in required_columns:
        if column not in train_dataset.column_names:
            raise ValueError(
                f"Dataset for {config.loss_type!r} is missing column {column!r}."
            )

    drop_columns = [
        column
        for column in train_dataset.column_names
        if column not in keep_columns
    ]
    if drop_columns:
        train_dataset = train_dataset.remove_columns(drop_columns)

    if config.loss_type in ("mnrl", "cached_mnrl") and "label" in train_dataset.column_names:
        train_dataset = train_dataset.filter(
            lambda row: int(row["label"]) == 1
        ).rename_columns(
            {
                "text_a": "anchor",
                "text_b": "positive",
            }
        ).remove_columns(["label"])
        if len(train_dataset) == 0:
            raise ValueError(
                f"No positive pairs remain for {config.loss_type!r} training."
            )

    if config.eval_ratio == 0.0:
        return datasets.DatasetDict({"train": train_dataset})

    split_dataset = train_dataset.train_test_split(
        test_size=config.eval_ratio,
        seed=config.random_seed,
        shuffle=True,
    )
    return datasets.DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        }
    )


def load_model(config: TrainConfig) -> sentence_transformers.SentenceTransformer:
    """Load the base embedding model as a SentenceTransformer."""
    model_kwargs: dict[str, typing.Any] = {}
    if config.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if config.load_in_4bit or config.load_in_8bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )
        model_kwargs["device_map"] = "auto"

    return sentence_transformers.SentenceTransformer(
        config.model_name,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=model_kwargs,
        trust_remote_code=config.trust_remote_code,
    )


def apply_lora(
    config: TrainConfig,
    model: sentence_transformers.SentenceTransformer,
) -> sentence_transformers.SentenceTransformer:
    """Attach PEFT LoRA adapters to the encoder backbone."""
    transformer_module = typing.cast(typing.Any, model[0])
    base_model = typing.cast(typing.Any, transformer_module.auto_model)

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    peft_model = typing.cast(typing.Any, peft.get_peft_model(base_model, lora_config))
    transformer_module.auto_model = peft_model
    transformer_module.auto_model.config.use_cache = False
    if hasattr(transformer_module.auto_model, "gradient_checkpointing_enable"):
        transformer_module.auto_model.gradient_checkpointing_enable()
    return model


def build_training_args(
    config: TrainConfig,
    has_validation: bool,
) -> sentence_transformers.training_args.SentenceTransformerTrainingArguments:
    """Build trainer arguments."""
    max_steps = config.max_steps if config.max_steps > 0 else -1
    warmup_steps = config.warmup_ratio
    batch_sampler = (
        sentence_transformers.training_args.BatchSamplers.NO_DUPLICATES
        if config.loss_type in ("mnrl", "cached_mnrl")
        else sentence_transformers.training_args.BatchSamplers.BATCH_SAMPLER
    )
    return sentence_transformers.training_args.SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=config.save_steps if has_validation else None,
        save_total_limit=config.save_total_limit,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="wandb",
        lr_scheduler_type="constant_with_warmup",
        seed=config.random_seed,
        batch_sampler=batch_sampler,
    )


def build_loss(
    config: TrainConfig,
    model: sentence_transformers.SentenceTransformer,
) -> typing.Any:
    """Build the configured training loss."""
    if config.loss_type == "triplet":
        return sentence_transformers.losses.TripletLoss(
            model=model,
            triplet_margin=config.triplet_margin,
        )
    if config.loss_type == "mnrl":
        return sentence_transformers.losses.MultipleNegativesRankingLoss(
            model=model,
        )
    if config.loss_type == "cached_mnrl":
        return sentence_transformers.losses.CachedMultipleNegativesRankingLoss(
            model=model,
        )
    raise ValueError(f"Unsupported loss type: {config.loss_type!r}")


def train_model(config: TrainConfig) -> None:
    """Run LoRA fine-tuning."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_dataset_for_loss(config)
    model = load_model(config)
    model = apply_lora(config, model)

    loss = build_loss(config, model)
    has_validation = "validation" in dataset_dict
    training_args = build_training_args(config, has_validation)

    trainer = sentence_transformers.SentenceTransformerTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        loss=loss,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir / "final_model"))


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    config = build_config(args)

    if not config.train_path.exists():
        raise FileNotFoundError(f"Triplet dataset not found: {config.train_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    train_model(config)
    print(f"saved trained model to {config.output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
