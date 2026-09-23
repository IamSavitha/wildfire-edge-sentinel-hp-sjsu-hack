"""LoRA-distill teacher context labels (scripts/teacher_label.py) into Qwen2.5-VL-7B. Run on the Nano.

Writes a PEFT adapter to --out; serve it with vLLM `--enable-lora --lora-modules context=<out>`
and score it with scripts/eval_context.py (plan Task 24 / 24b).

Fallback if TRL/transformers versions disagree: adapt HP's Med_VLM_Fine-Tune_vLLM script
(hack guide p.6) to this JSONL; the data format below is the same chat format.
"""
import argparse
import json

from sentinel.vlm_client import SYSTEM_PROMPT, USER_PROMPT  # training prompt == runtime prompt


def to_example(row: dict) -> dict:
    """One JSONL row -> TRL vision chat example (messages + images, one image placeholder)."""
    from PIL import Image  # lazy: --help works without pillow

    with Image.open(row["image"]) as im:
        image = im.convert("RGB")
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(row["label"])}]},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train", default="data/teacher/train.jsonl")
    ap.add_argument("--out", default="adapters/context")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank; vLLM needs --max-lora-rank >= this")
    ap.add_argument("--batch", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=448 * 448, help="processor image budget (pixels)")
    a = ap.parse_args()

    # lazy: --help and unit tests work without torch/transformers/trl/peft
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    with open(a.train) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ds = Dataset.from_list([to_example(r) for r in rows])
    print(f"{len(ds)} training examples from {a.train}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(a.base, max_pixels=a.max_pixels)
    peft_config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])  # language model only
    args = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs,
                     per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                     learning_rate=a.lr, bf16=True, logging_steps=10, save_strategy="epoch",
                     gradient_checkpointing=True, max_length=None, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=processor, peft_config=peft_config)
    trainer.train()
    trainer.save_model(a.out)
    print(f"adapter saved to {a.out}")


if __name__ == "__main__":
    main()
