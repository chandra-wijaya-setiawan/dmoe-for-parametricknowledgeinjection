"""Trains individual LoRA experts on knowledge units."""

from __future__ import annotations

import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


class ExpertTrainer:
    """Trains a LoRA adapter on final-layer FFN for one knowledge unit."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

    def train(
        self,
        knowledge_text: str,
        qa_pairs: list[dict],
        output_dir: str,
        lora_targets: list[str],
    ) -> str:
        """
        Train a LoRA expert on QA pairs derived from a knowledge passage.

        Args:
            knowledge_text: the source knowledge unit/passage
            qa_pairs: list of {"question": ..., "answer": ...}
            output_dir: directory to save the adapter
            lora_targets: module names for final-layer FFN LoRA
        """
        os.makedirs(output_dir, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"]["name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        for param in model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=self.config["expert"]["r"],
            lora_alpha=self.config["expert"]["alpha"],
            target_modules=lora_targets,
            lora_dropout=self.config["expert"]["dropout"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"]["name"])
        tokenizer.pad_token = tokenizer.eos_token

        formatted = []
        for qa in qa_pairs:
            messages = [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted.append({"text": text})

        dataset = Dataset.from_list(formatted)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["expert"]["epochs"],
            per_device_train_batch_size=self.config["expert"]["batch_size"],
            learning_rate=self.config["expert"]["learning_rate"],
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            fp16=(self.device == "cuda"),
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        model.save_pretrained(output_dir)

        del model, trainer
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return output_dir
