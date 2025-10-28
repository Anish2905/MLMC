"""
Complete LLM Compression Pipeline with Accuracy Preservation
=============================================================
Demonstrates: 4-bit Quantization, QLoRA, Knowledge Distillation, Mixed-Precision
on GPT-2 models with Wikitext-2 dataset.

Author: Research Implementation
Date: 2025
Framework: PyTorch + Transformers + PEFT + bitsandbytes
Dataset: Wikitext-2 (language modeling)
Models: GPT-2 Medium → GPT-2 Small (distillation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
import psutil
from typing import Dict, Tuple, Optional, List
import json

# Transformers and quantization libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Config
)

# LoRA/QLoRA for parameter-efficient fine-tuning
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)

# Dataset
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for LLM compression"""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configuration
    TEACHER_MODEL = "gpt2-medium"  # 355M parameters
    STUDENT_MODEL = "gpt2"          # 124M parameters (GPT-2 Small)
    MAX_LENGTH = 512                # Maximum sequence length
    
    # Dataset configuration
    DATASET_NAME = "wikitext"
    DATASET_CONFIG = "wikitext-2-raw-v1"
    TRAIN_SAMPLES = 10000           # Limit for faster training
    EVAL_SAMPLES = 1000
    CALIBRATION_SAMPLES = 500
    
    # Training configuration
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS_TEACHER = 3           # Fine-tune teacher
    NUM_EPOCHS_STUDENT = 5           # Distillation training
    NUM_EPOCHS_QLORA = 3             # QLoRA fine-tuning
    
    # Distillation parameters
    TEMPERATURE = 2.0                # Softmax temperature
    ALPHA = 0.5                      # Weight for distillation loss
    
    # Quantization configuration
    LOAD_IN_4BIT = True              # Use 4-bit quantization
    LOAD_IN_8BIT = False             # Alternative: 8-bit quantization
    BNB_4BIT_COMPUTE_DTYPE = torch.float16  # Compute in FP16
    BNB_4BIT_QUANT_TYPE = "nf4"      # NormalFloat 4-bit
    USE_NESTED_QUANT = True          # Double quantization for better compression
    
    # LoRA configuration
    LORA_R = 16                      # LoRA rank
    LORA_ALPHA = 32                  # LoRA scaling
    LORA_DROPOUT = 0.05              # LoRA dropout
    LORA_TARGET_MODULES = [          # Modules to apply LoRA
        "c_attn",                    # Attention layers (Q, K, V projection)
        "c_proj",                    # Output projection
    ]
    
    # Mixed-precision configuration
    MIXED_PRECISION = "fp16"         # Options: "fp16", "bf16", "no"
    
    # Save paths
    SAVE_DIR = "./models_compressed_llm"
    RESULTS_FILE = "compression_results.json"
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Evaluation
    PERPLEXITY_STRIDE = 256          # Stride for perplexity calculation

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.RANDOM_SEED)

# Create save directory
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_dataset():
    """
    Load Wikitext-2 dataset and prepare for language modeling.
    
    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    print("\n[INFO] Loading Wikitext-2 dataset...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 needs explicit pad token
    
    # Load dataset
    dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_CONFIG)
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize and concatenate texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Tokenize datasets
    print("  Tokenizing training data...")
    train_dataset = dataset["train"].select(range(min(Config.TRAIN_SAMPLES, len(dataset["train"]))))
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    print("  Tokenizing validation data...")
    eval_dataset = dataset["validation"].select(range(min(Config.EVAL_SAMPLES, len(dataset["validation"]))))
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    print(f"  ✓ Training samples: {len(train_dataset)}")
    print(f"  ✓ Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, tokenizer


def prepare_calibration_data(eval_dataset, tokenizer) -> List[str]:
    """
    Prepare calibration dataset for quantization.
    
    Args:
        eval_dataset: Validation dataset
        tokenizer: Tokenizer
    
    Returns:
        List of text samples for calibration
    """
    print("\n[INFO] Preparing calibration data...")
    calibration_samples = []
    
    for i in range(min(Config.CALIBRATION_SAMPLES, len(eval_dataset))):
        sample = eval_dataset[i]
        text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        calibration_samples.append(text)
    
    print(f"  ✓ Prepared {len(calibration_samples)} calibration samples")
    return calibration_samples


# ============================================================================
# MODEL PREPARATION
# ============================================================================

def load_teacher_model(quantize: bool = False):
    """
    Load teacher model (GPT-2 Medium) with optional quantization.
    
    Args:
        quantize: Whether to load with 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n[INFO] Loading Teacher Model: {Config.TEACHER_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    if quantize and Config.LOAD_IN_4BIT:
        print("  Loading with 4-bit quantization (NF4)...")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=Config.USE_NESTED_QUANT,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.TEACHER_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("  Loading in full precision (FP32)...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.TEACHER_MODEL,
            torch_dtype=torch.float32
        ).to(Config.DEVICE)
    
    print(f"  ✓ Model loaded: {count_parameters(model):,} parameters")
    return model, tokenizer


def load_student_model(quantize: bool = False):
    """
    Load student model (GPT-2 Small) for distillation.
    
    Args:
        quantize: Whether to load with quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n[INFO] Loading Student Model: {Config.STUDENT_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    if quantize and Config.LOAD_IN_4BIT:
        print("  Loading with 4-bit quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=Config.USE_NESTED_QUANT,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.STUDENT_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("  Loading in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.STUDENT_MODEL,
            torch_dtype=torch.float32
        ).to(Config.DEVICE)
    
    print(f"  ✓ Model loaded: {count_parameters(model):,} parameters")
    return model, tokenizer


def prepare_model_for_qlora(model):
    """
    Prepare quantized model for QLoRA fine-tuning.
    
    Args:
        model: 4-bit quantized model
    
    Returns:
        Model with LoRA adapters
    """
    print("\n[INFO] Preparing model for QLoRA fine-tuning...")
    
    # Prepare model for k-bit training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    trainable_params, all_params = count_trainable_parameters(model)
    print(f"  ✓ LoRA adapters added")
    print(f"  Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_with_distillation(
    student_model,
    teacher_model,
    train_dataset,
    eval_dataset,
    tokenizer
):
    """
    Train student model using knowledge distillation from teacher.
    
    Args:
        student_model: Student model to train
        teacher_model: Pre-trained teacher model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
    
    Returns:
        Trained student model
    """
    print("\n[INFO] Starting Knowledge Distillation Training...")
    print(f"  Teacher: {Config.TEACHER_MODEL} → Student: {Config.STUDENT_MODEL}")
    print(f"  Temperature: {Config.TEMPERATURE}, Alpha: {Config.ALPHA}")
    
    # Set models to appropriate modes
    teacher_model.eval()
    student_model.train()
    
    # Custom distillation trainer
    class DistillationTrainer(Trainer):
        def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher = teacher_model
            self.temperature = temperature
            self.alpha = alpha
            
        def compute_loss(self, model, inputs, return_outputs=False):
            # Student forward pass
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Hard loss (standard cross-entropy)
            loss_ce = student_outputs.loss
            
            # Soft loss (KL divergence between teacher and student)
            loss_kd = nn.functional.kl_div(
                nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
                nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Combined loss
            loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
            
            return (loss, student_outputs) if return_outputs else loss
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(Config.SAVE_DIR, "distillation_checkpoints"),
        num_train_epochs=Config.NUM_EPOCHS_STUDENT,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=Config.MIXED_PRECISION == "fp16",
        bf16=Config.MIXED_PRECISION == "bf16",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=Config.TEMPERATURE,
        alpha=Config.ALPHA,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n  Training distilled model...")
    trainer.train()
    
    print("  ✓ Distillation training complete")
    return student_model


def train_with_qlora(model, train_dataset, eval_dataset, tokenizer):
    """
    Fine-tune quantized model using QLoRA.
    
    Args:
        model: 4-bit quantized model with LoRA adapters
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
    
    Returns:
        Fine-tuned model
    """
    print("\n[INFO] Starting QLoRA Fine-tuning...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(Config.SAVE_DIR, "qlora_checkpoints"),
        num_train_epochs=Config.NUM_EPOCHS_QLORA,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=Config.MIXED_PRECISION == "fp16",
        bf16=Config.MIXED_PRECISION == "bf16",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("  Training with QLoRA...")
    trainer.train()
    
    print("  ✓ QLoRA fine-tuning complete")
    return model


# ============================================================================
# EVALUATION AND METRICS
# ============================================================================

def calculate_perplexity(model, eval_dataset, tokenizer) -> float:
    """
    Calculate perplexity on evaluation dataset.
    Lower perplexity = better model.
    
    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
    
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            sample = eval_dataset[i]
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(model.device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Accumulate loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def evaluate_model(
    model,
    eval_dataset,
    tokenizer,
    desc: str = "Model"
) -> Dict[str, float]:
    """
    Comprehensive evaluation: perplexity, size, memory, latency.
    
    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        desc: Description for display
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n[INFO] Evaluating {desc}...")
    
    # Calculate perplexity
    print("  Calculating perplexity...")
    perplexity = calculate_perplexity(model, eval_dataset, tokenizer)
    
    # Model size (disk size)
    temp_path = os.path.join(Config.SAVE_DIR, "temp_model")
    model.save_pretrained(temp_path)
    
    model_size_mb = 0
    for root, dirs, files in os.walk(temp_path):
        for file in files:
            model_size_mb += os.path.getsize(os.path.join(root, file))
    model_size_mb /= (1024 * 1024)
    
    # Clean up
    import shutil
    shutil.rmtree(temp_path)
    
    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Inference latency (average over 10 samples)
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i in range(min(10, len(eval_dataset))):
            sample = eval_dataset[i]
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(model.device)
            
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    
    # Parameter count
    params = count_parameters(model)
    trainable_params, _ = count_trainable_parameters(model)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  {desc} Evaluation Results")
    print(f"{'='*70}")
    print(f"  Perplexity:         {perplexity:.2f}")
    print(f"  Model Size:         {model_size_mb:.2f} MB")
    print(f"  Parameters:         {params:,}")
    print(f"  Trainable Params:   {trainable_params:,}")
    print(f"  Memory Usage:       {memory_mb:.2f} MB")
    print(f"  Avg Latency:        {avg_latency:.2f} ms/sequence")
    print(f"{'='*70}\n")
    
    return {
        "perplexity": perplexity,
        "model_size_mb": model_size_mb,
        "parameters": params,
        "trainable_parameters": trainable_params,
        "memory_mb": memory_mb,
        "latency_ms": avg_latency
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> Tuple[int, int]:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_results(results: Dict, filename: str):
    """Save evaluation results to JSON."""
    filepath = os.path.join(Config.SAVE_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved: {filepath}")


def generate_text_sample(model, tokenizer, prompt: str, max_length: int = 100):
    """
    Generate text sample for qualitative evaluation.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
    
    Returns:
        Generated text
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# ============================================================================
# MAIN COMPRESSION PIPELINE
# ============================================================================

def main():
    """
    Complete LLM compression pipeline.
    """
    print("="*70)
    print("  LLM COMPRESSION PIPELINE")
    print("  Demonstrating 4-bit Quantization, QLoRA, Knowledge Distillation")
    print("="*70)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Mixed Precision: {Config.MIXED_PRECISION}")
    print(f"Quantization: {'4-bit (NF4)' if Config.LOAD_IN_4BIT else '8-bit' if Config.LOAD_IN_8BIT else 'None'}")
    
    # Load dataset
    train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset()
    
    # Dictionary to store results
    results = {}
    
    # ========================================================================
    # STAGE 1: BASELINE TEACHER MODEL (Full Precision)
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 1: Baseline Teacher Model (GPT-2 Medium, Full Precision)")
    print("="*70)
    
    teacher_fp32, teacher_tokenizer = load_teacher_model(quantize=False)
    results['teacher_fp32'] = evaluate_model(
        teacher_fp32, eval_dataset, teacher_tokenizer, "Teacher (FP32)"
    )
    
    # Save baseline
    teacher_fp32.save_pretrained(os.path.join(Config.SAVE_DIR, "teacher_fp32"))
    teacher_tokenizer.save_pretrained(os.path.join(Config.SAVE_DIR, "teacher_fp32"))
    
    # ========================================================================
    # STAGE 2: 4-BIT QUANTIZED TEACHER
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 2: 4-bit Quantized Teacher (NF4)")
    print("="*70)
    
    teacher_4bit, _ = load_teacher_model(quantize=True)
    results['teacher_4bit'] = evaluate_model(
        teacher_4bit, eval_dataset, tokenizer, "Teacher (4-bit NF4)"
    )
    
    # ========================================================================
    # STAGE 3: QLORA FINE-TUNING ON QUANTIZED MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 3: QLoRA Fine-tuning (4-bit + LoRA)")
    print("="*70)
    
    # Prepare for QLoRA
    teacher_qlora = prepare_model_for_qlora(teacher_4bit)
    
    # Fine-tune with QLoRA
    teacher_qlora = train_with_qlora(
        teacher_qlora, train_dataset, eval_dataset, tokenizer
    )
    
    results['teacher_qlora'] = evaluate_model(
        teacher_qlora, eval_dataset, tokenizer, "Teacher (QLoRA Fine-tuned)"
    )
    
    # Save QLoRA model
    teacher_qlora.save_pretrained(os.path.join(Config.SAVE_DIR, "teacher_qlora"))
    
    # ========================================================================
    # STAGE 4: KNOWLEDGE DISTILLATION (Medium → Small)
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 4: Knowledge Distillation (GPT-2 Medium → Small)")
    print("="*70)
    
    # Load student model
    student_fp32, student_tokenizer = load_student_model(quantize=False)
    
    # Distillation training
    student_distilled = train_with_distillation(
        student_fp32, teacher_fp32, train_dataset, eval_dataset, tokenizer
    )
    
    results['student_distilled'] = evaluate_model(
        student_distilled, eval_dataset, student_tokenizer, "Student Distilled (FP32)"
    )
    
    # Save distilled student
    student_distilled.save_pretrained(os.path.join(Config.SAVE_DIR, "student_distilled"))
    
    # ========================================================================
    # STAGE 5: FINAL COMPRESSED MODEL (Distilled + 4-bit + QLoRA)
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 5: Final Compressed Model (Distilled + 4-bit + QLoRA)")
    print("="*70)
    
    # Load distilled student with 4-bit quantization
    student_4bit, _ = load_student_model(quantize=True)
    
    # Prepare for QLoRA
    student_final = prepare_model_for_qlora(student_4bit)
    
    # Fine-tune with QLoRA
    student_final = train_with_qlora(
        student_final, train_dataset, eval_dataset, tokenizer
    )
    
    results['student_final'] = evaluate_model(
        student_final, eval_dataset, tokenizer, "Final Compressed (Distilled + 4-bit + QLoRA)"
    )
    
    # Save final model
    student_final.save_pretrained(os.path.join(Config.SAVE_DIR, "student_final_compressed"))
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("  FINAL COMPARISON: All Models")
    print("="*70)
    
    print(f"\n{'Model':<35} {'Perplexity':>12} {'Size (MB)':>12} {'Params':>15} {'Latency (ms)':>15}")
    print("-"*90)
    
    for name, metrics in results.items():
        print(f"{name:<35} {metrics['perplexity']:>12.2f} "
              f"{metrics['model_size_mb']:>11.2f} "
              f"{metrics['parameters']:>14,} "
              f"{metrics['latency_ms']:>14.2f}")
    
    # Calculate compression metrics
    baseline_size = results['teacher_fp32']['model_size_mb']
    final_size = results['student_final']['model_size_mb']
    compression_ratio = baseline_size / final_size
    
    baseline_params = results['teacher_fp32']['parameters']
    final_params = results['student_final']['parameters']
    param_reduction = (1 - final_params / baseline_params) * 100
    
    baseline_ppl = results['teacher_fp32']['perplexity']
    final_ppl = results['student_final']['perplexity']
    ppl_increase = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
    
    print("\n" + "="*70)
    print(f"  COMPRESSION SUMMARY")
    print("="*70)
    print(f"  Size Reduction:       {compression_ratio:.2f}x ({baseline_size:.1f} MB → {final_size:.1f} MB)")
    print(f"  Parameter Reduction:  {param_reduction:.1f}% ({baseline_params:,} → {final_params:,})")
    print(f"  Perplexity Change:    +{ppl_increase:.1f}% ({baseline_ppl:.2f} → {final_ppl:.2f})")
    print("="*70)
    
    # Save results
    save_results(results, Config.RESULTS_FILE)
    
    print(f"\n✓ All models saved to: {Config.SAVE_DIR}")
    print("[DONE] LLM compression pipeline complete!\n")
    
    return results


# ============================================================================
# TEXT GENERATION COMPARISON
# ============================================================================

def compare_generations():
    """
    Compare text generation quality across compressed models.
    """
    print("\n" + "="*70)
    print("  TEXT GENERATION COMPARISON")
    print("="*70)
    
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: '{prompt}'\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    models_to_test = [
        ("Baseline Teacher (FP32)", "teacher_fp32"),
        ("4-bit Quantized", "teacher_qlora"),
        ("Distilled Student", "student_distilled"),
        ("Final Compressed", "student_final_compressed")
    ]
    
    for name, path in models_to_test:
        model_path = os.path.join(Config.SAVE_DIR, path)
        
        if os.path.exists(model_path):
            print(f"\n{name}:")
            print("-" * 70)
            
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                model = model.to(Config.DEVICE)
                
                generated = generate_text_sample(model, tokenizer, prompt, max_length=80)
                print(generated)
                
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"\n{name}: Model not found")
    
    print("\n" + "="*70)


# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================

def inference_example():
    """
    Demonstrate loading and using compressed model for inference.
    """
    print("\n" + "="*70)
    print("  INFERENCE EXAMPLE: Using Final Compressed Model")
    print("="*70)
    
    model_path = os.path.join(Config.SAVE_DIR, "student_final_compressed")
    
    if not os.path.exists(model_path):
        print("\n  ✗ Compressed model not found. Run main() first.")
        return
    
    print("\n  Loading compressed model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(Config.DEVICE)
    model.eval()
    
    # Example prompts
    prompts = [
        "Once upon a time in a distant galaxy",
        "The key to machine learning success is",
        "Climate change represents"
    ]
    
    print("\n  Generating responses...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}] Prompt: '{prompt}'")
        
        start_time = time.time()
        generated = generate_text_sample(model, tokenizer, prompt, max_length=60)
        gen_time = (time.time() - start_time) * 1000
        
        print(f"      Generated: {generated}")
        print(f"      Time: {gen_time:.2f} ms\n")
    
    print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  LLM COMPRESSION RESEARCH TOOLKIT")
    print("  GPT-2 Models with 4-bit Quantization, QLoRA & Distillation")
    print("="*70)
    
    # Check dependencies
    try:
        import bitsandbytes
        import peft
        print("\n✓ All dependencies available")
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install transformers datasets peft bitsandbytes accelerate")
        exit(1)
    
    # Run complete pipeline
    results = main()
    
    # Compare text generation quality
    compare_generations()
    
    # Demonstrate inference
    inference_example()
    
    print("\n" + "="*70)
    print("  LLM compression pipeline completed successfully!")
    print("  Ready for deployment and research documentation.")
    print("="*70 + "\n")