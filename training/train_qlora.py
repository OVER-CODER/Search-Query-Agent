#!/usr/bin/env python3
"""
QLoRA fine-tuning script for 7B model on browser history data.
Uses 4-bit quantization and LoRA for memory efficiency.
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb
from tqdm import tqdm


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(
        default="microsoft/DialoGPT-medium",  # Fallback for testing
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout parameter"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA r parameter"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias parameter"}
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "LoRA task type"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_path: str = field(
        default="training_data/train.jsonl",
        metadata={"help": "Path to training dataset"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


class QLoRATrainer:
    """QLoRA training pipeline."""
    
    def __init__(self, model_args: ModelArguments, lora_args: LoraArguments, data_args: DataArguments):
        self.model_args = model_args
        self.lora_args = lora_args
        self.data_args = data_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=self.model_args.use_4bit,
            bnb_4bit_use_double_quant=self.model_args.use_nested_quant,
            bnb_4bit_quant_type=self.model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.model_args.bnb_4bit_compute_dtype)
        )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization."""
        print(f"Loading model: {self.model_args.model_name_or_path}")
        
        # Create quantization config
        bnb_config = self.create_bnb_config()
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                quantization_config=bnb_config if self.model_args.use_4bit else None,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            print(f"Failed to load {self.model_args.model_name_or_path}: {e}")
            print("Falling back to smaller model for testing...")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",
                quantization_config=bnb_config if self.model_args.use_4bit else None,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def prepare_model_for_training(self, model):
        """Prepare model for LoRA training."""
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"] if "llama" in self.model_args.model_name_or_path.lower() 
                         else ["c_attn", "c_proj"],  # For DialoGPT and similar
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def load_dataset(self, tokenizer) -> Dataset:
        """Load and tokenize dataset."""
        print(f"Loading dataset from: {self.data_args.dataset_path}")
        
        # Read JSONL file
        data = []
        try:
            with open(self.data_args.dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print("No training data found. Creating synthetic dataset for testing...")
            data = self.create_synthetic_dataset()
        
        print(f"Loaded {len(data)} training examples")
        
        # Format for instruction tuning
        formatted_data = []
        for example in data:
            instruction = example["instruction"]
            response = example["response"]
            
            # Format as conversation
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}<|endoftext|>"
            formatted_data.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.data_args.max_length,
                return_tensors=None,
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def create_synthetic_dataset(self) -> list:
        """Create synthetic dataset for testing."""
        synthetic_data = [
            {
                "instruction": "Given query: 'python tutorial', which URL from my browsing history best answers it?",
                "response": '{"url": "https://docs.python.org/3/tutorial/", "title": "Python Tutorial", "domain": "docs.python.org", "confidence": 0.9, "reason": "You previously visited this page 15 times"}'
            },
            {
                "instruction": "Given query: 'machine learning', which URL from my browsing history best answers it?",
                "response": '{"url": "https://scikit-learn.org/stable/", "title": "scikit-learn Documentation", "domain": "scikit-learn.org", "confidence": 0.8, "reason": "You previously visited this page 8 times"}'
            },
            {
                "instruction": "Given query: 'fastapi documentation', which URL from my browsing history best answers it?",
                "response": '{"url": "https://fastapi.tiangolo.com/", "title": "FastAPI", "domain": "fastapi.tiangolo.com", "confidence": 0.95, "reason": "You previously visited this page 12 times"}'
            }
        ]
        
        # Save synthetic dataset
        os.makedirs("training_data", exist_ok=True)
        with open("training_data/train.jsonl", 'w') as f:
            for example in synthetic_data:
                f.write(json.dumps(example) + '\n')
        
        return synthetic_data
    
    def train(self, training_args: TrainingArguments):
        """Main training function."""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare model for training
        model = self.prepare_model_for_training(model)
        
        # Load dataset
        train_dataset = self.load_dataset(tokenizer)
        
        # Create trainer
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        
        return model, tokenizer


def main():
    """Main training script."""
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, TrainingArguments))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set default training arguments if not provided
    if not hasattr(training_args, 'output_dir') or training_args.output_dir is None:
        training_args.output_dir = "./checkpoints"
    if not hasattr(training_args, 'num_train_epochs') or training_args.num_train_epochs is None:
        training_args.num_train_epochs = 1
    if not hasattr(training_args, 'per_device_train_batch_size') or training_args.per_device_train_batch_size is None:
        training_args.per_device_train_batch_size = 1
    if not hasattr(training_args, 'learning_rate') or training_args.learning_rate is None:
        training_args.learning_rate = 2e-4
    if not hasattr(training_args, 'max_steps') or training_args.max_steps is None:
        training_args.max_steps = 10  # Small for testing
    
    # Additional training arguments
    training_args.gradient_accumulation_steps = 4
    training_args.optim = "paged_adamw_32bit"
    training_args.save_steps = 5
    training_args.logging_steps = 1
    training_args.warmup_ratio = 0.03
    training_args.group_by_length = True
    training_args.lr_scheduler_type = "constant"
    training_args.report_to = []  # Disable wandb
    
    # Create trainer
    trainer = QLoRATrainer(model_args, lora_args, data_args)
    
    # Train
    model, tokenizer = trainer.train(training_args)
    
    print("Training complete!")
    print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()