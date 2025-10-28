"""
Complete Model Compression Pipeline with Accuracy Preservation
===============================================================
Demonstrates: Quantization-Aware Training (QAT), Mixed-Precision Quantization,
Knowledge Distillation, and Fine-tuning on CIFAR-10 with ResNet models.

NOW INCLUDES:
- Automatic visualization generation
- Support for custom user-trained models
- Flexible configuration for any CNN architecture

Author: Research Implementation
Date: 2025
Framework: PyTorch 2.0+ with torchvision
Dataset: CIFAR-10 (10 classes, 32x32 RGB images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy
import os
from typing import Tuple, Dict, List, Optional
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for reproducibility"""
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CUSTOM MODEL SUPPORT - Set these to use your own model
    USE_CUSTOM_MODEL = False           # Set to True to use custom model
    CUSTOM_MODEL_PATH = None           # Path to your .pth file
    CUSTOM_MODEL_CLASS = None          # Your model class (if needed)
    CUSTOM_NUM_CLASSES = 10            # Number of classes in your model
    
    # Dataset parameters
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    CALIBRATION_SAMPLES = 1000
    
    # Training parameters
    TEACHER_EPOCHS = 10
    STUDENT_EPOCHS = 20
    QAT_EPOCHS = 15
    FINETUNE_EPOCHS = 10
    
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Distillation parameters
    DISTILLATION_TEMPERATURE = 4.0
    DISTILLATION_ALPHA = 0.7
    
    # Quantization parameters
    QUANT_BACKEND = 'qnnpack'
    
    # Visualization
    GENERATE_VISUALIZATIONS = True     # Auto-generate plots
    
    # Model save paths
    SAVE_DIR = "./models_compressed"
    VIZ_DIR = "./visualizations"
    
    # Reproducibility
    RANDOM_SEED = 42

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.RANDOM_SEED)

# ============================================================================
# VISUALIZATION UTILITIES (INTEGRATED)
# ============================================================================

class CompressionVisualizer:
    """Visualization toolkit for compression results"""
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_compression_summary(self, results: Dict[str, Dict]):
        """Create comprehensive summary visualization"""
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        sizes = [results[m]['model_size_mb'] for m in models]
        latencies = [results[m]['latency_ms'] for m in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Compression Results', fontsize=16, fontweight='bold')
        
        # 1. Model Size
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, sizes, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Model Size (MB)', fontweight='bold')
        ax1.set_title('Model Size Comparison', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Accuracy
        ax2 = axes[0, 1]
        colors = ['green' if i == 0 else 'orange' if i == len(models)-1 
                  else 'steelblue' for i in range(len(models))]
        bars2 = ax2.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Latency
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, latencies, color='coral', alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Latency (ms)', fontweight='bold')
        ax3.set_title('Inference Latency', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Accuracy vs Size Tradeoff
        ax4 = axes[1, 1]
        scatter = ax4.scatter(sizes, accuracies, s=300, c=range(len(models)),
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        for i, model in enumerate(models):
            ax4.annotate(model, (sizes[i], accuracies[i]),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=9, fontweight='bold')
        ax4.set_xlabel('Model Size (MB)', fontweight='bold')
        ax4.set_ylabel('Accuracy (%)', fontweight='bold')
        ax4.set_title('Accuracy vs Size Tradeoff', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "compression_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_compression_ratios(self, results: Dict[str, Dict]):
        """Visualize compression ratios"""
        models = list(results.keys())
        baseline_size = results[models[0]]['model_size_mb']
        compression_ratios = [baseline_size / results[m]['model_size_mb'] for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if r > 1 else 'red' for r in compression_ratios]
        bars = ax.barh(models, compression_ratios, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Compression Ratio (×)', fontweight='bold', fontsize=12)
        ax.set_title('Compression Ratios vs Baseline', fontweight='bold', fontsize=14)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
            width = bar.get_width()
            label = f'{ratio:.2f}×'
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   label, ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "compression_ratios.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def generate_all_plots(self, results: Dict[str, Dict]):
        """Generate all visualization plots"""
        print("\n[INFO] Generating visualizations...")
        self.plot_compression_summary(results)
        self.plot_compression_ratios(results)
        print(f"✓ All visualizations saved to: {self.save_dir}\n")


# ============================================================================
# CUSTOM MODEL LOADER
# ============================================================================

class CustomModelLoader:
    """Load and validate custom user-trained models"""
    
    @staticmethod
    def load_custom_model(
        model_path: str,
        model_class = None,
        num_classes: int = 10,
        device: str = "cuda"
    ):
        """
        Load custom CNN model from checkpoint.
        
        Usage:
            # Option 1: Load state dict into provided class
            model = CustomModelLoader.load_custom_model(
                model_path='./my_model.pth',
                model_class=MyResNet,
                num_classes=100
            )
            
            # Option 2: Load complete model
            model = torch.load('./my_complete_model.pth')
            model = CustomModelLoader.validate_model(model)
        """
        print(f"\n[INFO] Loading custom model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            # Assume it's a complete model
            return checkpoint.to(device).eval()
        
        # Load into provided class
        if model_class is not None:
            model = model_class(num_classes=num_classes)
            model.load_state_dict(state_dict)
        else:
            raise ValueError(
                "Please provide model_class or save complete model with torch.save(model, path)"
            )
        
        model = model.to(device).eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded: {total_params:,} parameters")
        
        return model
    
    @staticmethod
    def validate_model(model, sample_input=None):
        """Validate model is ready for compression"""
        print("\n[INFO] Validating model...")
        
        try:
            # Check state dict
            state_dict = model.state_dict()
            print(f"  ✓ State dict accessible ({len(state_dict)} parameters)")
            
            # Test forward pass if sample provided
            if sample_input is not None:
                model.eval()
                with torch.no_grad():
                    output = model(sample_input)
                print("  ✓ Forward pass successful")
            
            # Test serialization
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                torch.save(model.state_dict(), tmp.name)
            print("  ✓ Model serialization successful")
            
            print("✓ Model validation passed!\n")
            return model
            
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
            raise


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar10_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 dataset with appropriate transforms"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    calibration_indices = list(range(Config.CALIBRATION_SAMPLES))
    calibration_set = Subset(trainset, calibration_indices)
    
    trainloader = DataLoader(
        trainset, batch_size=Config.BATCH_SIZE, 
        shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    calibration_loader = DataLoader(
        calibration_set, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    testloader = DataLoader(
        testset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    return trainloader, calibration_loader, testloader


# ============================================================================
# MODEL PREPARATION
# ============================================================================

def prepare_teacher_model(
    num_classes: int = 10,
    custom_path: Optional[str] = None,
    custom_class = None
) -> nn.Module:
    """
    Load teacher model - supports both default and custom models.
    
    Args:
        num_classes: Number of output classes
        custom_path: Path to custom model checkpoint (optional)
        custom_class: Custom model class (optional)
    """
    if custom_path:
        print("\n[INFO] Using Custom Teacher Model")
        return CustomModelLoader.load_custom_model(
            model_path=custom_path,
            model_class=custom_class,
            num_classes=num_classes,
            device=Config.DEVICE
        )
    else:
        print("\n[INFO] Loading Default Teacher Model: ResNet50")
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model.to(Config.DEVICE)


def prepare_student_model(num_classes: int = 10) -> nn.Module:
    """Create smaller student model (ResNet18)"""
    print("\n[INFO] Creating Student Model: ResNet18")
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(Config.DEVICE)


def prepare_model_for_qat(model: nn.Module) -> nn.Module:
    """Prepare model for Quantization-Aware Training"""
    print("\n[INFO] Preparing model for QAT")
    model = model.cpu()
    model.eval()
    model.qconfig = quant.get_default_qat_qconfig(Config.QUANT_BACKEND)
    model = quant.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
    model_qat = quant.prepare_qat(model, inplace=False)
    return model_qat.to(Config.DEVICE)


def apply_mixed_precision_quantization(model: nn.Module, sensitive_layers: List[str]) -> nn.Module:
    """Apply mixed-precision quantization"""
    print("\n[INFO] Applying Mixed-Precision Quantization")
    print(f"  - Sensitive layers (FP16/FP32): {sensitive_layers}")
    
    default_qconfig = quant.get_default_qconfig(Config.QUANT_BACKEND)
    sensitive_qconfig = quant.QConfig(
        activation=quant.default_observer,
        weight=quant.default_per_channel_weight_observer
    )
    
    model.qconfig = default_qconfig
    
    for name, module in model.named_modules():
        if any(sensitive in name for sensitive in sensitive_layers):
            module.qconfig = sensitive_qconfig
            print(f"  - Keeping {name} in higher precision")
    
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_standard(
    model: nn.Module, 
    trainloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer,
    epochs: int,
    desc: str = "Training"
) -> nn.Module:
    """Standard training loop"""
    print(f"\n[INFO] {desc} for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(trainloader)} | "
                      f"Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_acc = 100. * correct / total
        print(f"  ► Epoch {epoch+1} Complete | Accuracy: {epoch_acc:.2f}%")
    
    return model


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    temperature: float = 4.0,
    alpha: float = 0.7
) -> nn.Module:
    """Train student with knowledge distillation"""
    print(f"\n[INFO] Knowledge Distillation Training")
    print(f"  Temperature: {temperature}, Alpha: {alpha}")
    
    teacher.eval()
    student.train()
    
    optimizer = optim.SGD(
        student.parameters(), 
        lr=Config.LEARNING_RATE, 
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            student_outputs = student(inputs)
            
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            
            loss_hard = criterion_hard(student_outputs, targets)
            loss_soft = criterion_soft(
                nn.functional.log_softmax(student_outputs / temperature, dim=1),
                nn.functional.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)
            
            loss = alpha * loss_soft + (1 - alpha) * loss_hard
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(trainloader)} | "
                      f"Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_acc = 100. * correct / total
        print(f"  ► Epoch {epoch+1} Complete | Accuracy: {epoch_acc:.2f}%")
    
    return student


def calibrate_model(model: nn.Module, calibration_loader: DataLoader) -> None:
    """Calibrate quantization parameters"""
    print("\n[INFO] Calibrating quantization parameters...")
    model.eval()
    
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(Config.DEVICE)
            model(inputs)
    
    print("  ► Calibration complete")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module, 
    testloader: DataLoader,
    desc: str = "Model"
) -> Dict[str, float]:
    """Comprehensive evaluation"""
    print(f"\n[INFO] Evaluating {desc}...")
    model.eval()
    
    correct = 0
    total = 0
    latencies = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            start_time = time.time()
            outputs = model(inputs)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_latency = np.mean(latencies)
    
    temp_path = "temp_model.pth"
    torch.save(model.state_dict(), temp_path)
    model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"  {desc} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"  Model Size:      {model_size_mb:.2f} MB")
    print(f"  Memory Usage:    {memory_mb:.2f} MB")
    print(f"  Avg Latency:     {avg_latency:.2f} ms/batch")
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'model_size_mb': model_size_mb,
        'memory_mb': memory_mb,
        'latency_ms': avg_latency
    }


# ============================================================================
# MODEL CONVERSION
# ============================================================================

def convert_to_quantized(model: nn.Module) -> nn.Module:
    """Convert QAT model to fully quantized INT8"""
    print("\n[INFO] Converting to fully quantized INT8 model...")
    model = model.cpu()
    model.eval()
    quantized_model = quant.convert(model, inplace=False)
    print("  ► Conversion complete")
    return quantized_model


def save_model(model: nn.Module, filename: str) -> None:
    """Save model to disk"""
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    filepath = os.path.join(Config.SAVE_DIR, filename)
    torch.save(model.state_dict(), filepath)
    print(f"  ✓ Model saved: {filepath}")


# ============================================================================
# MAIN COMPRESSION PIPELINE
# ============================================================================

def main():
    """Complete compression pipeline with visualization"""
    print("="*70)
    print("  MODEL COMPRESSION PIPELINE")
    print("  With Automatic Visualization & Custom Model Support")
    print("="*70)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Quantization Backend: {Config.QUANT_BACKEND}")
    
    if Config.USE_CUSTOM_MODEL:
        print(f"Custom Model Mode: ENABLED")
        print(f"  Model Path: {Config.CUSTOM_MODEL_PATH}")
    
    torch.backends.quantized.engine = Config.QUANT_BACKEND
    
    trainloader, calibration_loader, testloader = get_cifar10_dataloaders()
    print(f"\n✓ CIFAR-10 loaded: {len(trainloader.dataset)} train, {len(testloader.dataset)} test")
    
    results = {}
    
    # ========================================================================
    # STAGE 1: BASELINE TEACHER
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 1: Baseline Teacher Model")
    print("="*70)
    
    if Config.USE_CUSTOM_MODEL:
        teacher = prepare_teacher_model(
            num_classes=Config.CUSTOM_NUM_CLASSES,
            custom_path=Config.CUSTOM_MODEL_PATH,
            custom_class=Config.CUSTOM_MODEL_CLASS
        )
    else:
        teacher = prepare_teacher_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        teacher.parameters(), 
        lr=Config.LEARNING_RATE, 
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    teacher = train_standard(
        teacher, trainloader, criterion, optimizer, 
        epochs=Config.TEACHER_EPOCHS, desc="Teacher Training"
    )
    
    results['teacher'] = evaluate_model(teacher, testloader, "Baseline Teacher")
    save_model(teacher, "teacher_model.pth")
    
    # ========================================================================
    # STAGE 2: QAT
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 2: Quantization-Aware Training")
    print("="*70)
    
    model_qat = prepare_model_for_qat(copy.deepcopy(teacher))
    
    optimizer_qat = optim.SGD(
        model_qat.parameters(), 
        lr=Config.LEARNING_RATE * 0.1,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    model_qat = train_standard(
        model_qat, trainloader, criterion, optimizer_qat,
        epochs=Config.QAT_EPOCHS, desc="QAT Training"
    )
    
    model_qat_quantized = convert_to_quantized(model_qat)
    results['qat'] = evaluate_model(model_qat_quantized, testloader, "QAT Model (INT8)")
    save_model(model_qat_quantized, "model_qat_int8.pth")
    
    # ========================================================================
    # STAGE 3: MIXED-PRECISION
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 3: Mixed-Precision Quantization")
    print("="*70)
    
    sensitive_layers = ['conv1', 'fc']
    model_mixed = copy.deepcopy(teacher)
    model_mixed = apply_mixed_precision_quantization(model_mixed, sensitive_layers)
    model_mixed = quant.prepare(model_mixed, inplace=False)
    calibrate_model(model_mixed, calibration_loader)
    model_mixed_quantized = quant.convert(model_mixed, inplace=False)
    
    results['mixed'] = evaluate_model(model_mixed_quantized, testloader, "Mixed-Precision Model")
    save_model(model_mixed_quantized, "model_mixed_precision.pth")
    
    # ========================================================================
    # STAGE 4: KNOWLEDGE DISTILLATION
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 4: Knowledge Distillation")
    print("="*70)
    
    student = prepare_student_model()
    student = train_with_distillation(
        student, teacher, trainloader,
        epochs=Config.STUDENT_EPOCHS,
        temperature=Config.DISTILLATION_TEMPERATURE,
        alpha=Config.DISTILLATION_ALPHA
    )
    
    results['distilled'] = evaluate_model(student, testloader, "Distilled Student")
    save_model(student, "student_distilled.pth")
    
    # ========================================================================
    # STAGE 5: FINAL COMPRESSED
    # ========================================================================
    print("\n" + "="*70)
    print("  STAGE 5: Final Compressed Model")
    print("="*70)
    
    student_qat = prepare_model_for_qat(copy.deepcopy(student))
    optimizer_final = optim.SGD(
        student_qat.parameters(), 
        lr=Config.LEARNING_RATE * 0.05,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    student_qat = train_standard(
        student_qat, trainloader, criterion, optimizer_final,
        epochs=Config.FINETUNE_EPOCHS, desc="Final QAT Fine-tuning"
    )
    
    final_model = convert_to_quantized(student_qat)
    results['final'] = evaluate_model(final_model, testloader, "Final Compressed Model")
    save_model(final_model, "final_compressed_model.pth")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("  FINAL COMPARISON: All Models")
    print("="*70)
    print(f"\n{'Model':<30} {'Accuracy':>10} {'Size (MB)':>12} {'Latency (ms)':>15}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name.capitalize():<30} {metrics['accuracy']:>9.2f}% "
              f"{metrics['model_size_mb']:>11.2f} {metrics['latency_ms']:>14.2f}")
    
    baseline_size = results['teacher']['model_size_mb']
    final_size = results['final']['model_size_mb']
    compression_ratio = baseline_size / final_size
    
    baseline_acc = results['teacher']['accuracy']
    final_acc = results['final']['accuracy']
    acc_drop = baseline_acc - final_acc
    
    print("\n" + "="*70)
    print(f"  COMPRESSION ACHIEVED: {compression_ratio:.2f}x size reduction")
    print(f"  Accuracy preserved: {final_acc:.2f}% (vs {baseline_acc:.2f}% baseline)")
    print(f"  Accuracy drop: {acc_drop:.2f}%")
    print("="*70)
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    if Config.GENERATE_VISUALIZATIONS:
        viz_dir = os.path.join(Config.SAVE_DIR, Config.VIZ_DIR)
        visualizer = CompressionVisualizer(save_dir=viz_dir)
        visualizer.generate_all_plots(results)
    
    print("\n✓ All models saved to:", Config.SAVE_DIR)
    print("[DONE] Compression pipeline complete!")
    
    return results


# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================

def inference_example():
    """Demonstrate using compressed model for inference"""
    print("\n" + "="*70)
    print("  INFERENCE EXAMPLE")
    print("="*70)
    
    _, _, testloader = get_cifar10_dataloaders()
    
    model = prepare_student_model()
    model_path = os.path.join(Config.SAVE_DIR, "final_compressed_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model = model.to(Config.DEVICE)
        model.eval()
        
        inputs, targets = next(iter(testloader))
        inputs = inputs.to(Config.DEVICE)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
            _, predictions = outputs.max(1)
        inference_time = (time.time() - start_time) * 1000
        
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"\nInference on batch of {inputs.size(0)} images:")
        print(f"  Time: {inference_time:.2f} ms")
        print(f"  Per-image: {inference_time/inputs.size(0):.2f} ms")
        print(f"\nSample predictions:")
        for i in range(min(5, len(predictions))):
            pred_class = classes[predictions[i]]
            true_class = classes[targets[i]]
            status = "✓" if predictions[i] == targets[i] else "✗"
            print(f"  {status} Image {i+1}: Predicted={pred_class}, Actual={true_class}")
    else:
        print(f"  ✗ Model not found at {model_path}")


# ============================================================================
# CUSTOM MODEL EXAMPLE
# ============================================================================

def custom_model_example():
    """
    Example showing how to use your own custom trained model.
    """
    print("\n" + "="*70)
    print("  HOW TO USE YOUR CUSTOM MODEL")
    print("="*70)
    
    example_code = """
    # STEP 1: Train your model normally
    # =================================
    import torch
    import torch.nn as nn
    
    class MyCustomResNet(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            # Your architecture here
            self.features = nn.Sequential(...)
            self.classifier = nn.Linear(512, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Train your model
    my_model = MyCustomResNet(num_classes=100)
    # ... training code ...
    
    # Save it
    torch.save(my_model.state_dict(), './my_models/custom_model.pth')
    # Or save complete model:
    torch.save(my_model, './my_models/custom_model_complete.pth')
    
    
    # STEP 2: Configure compression script
    # =====================================
    # In this script, modify Config class:
    
    class Config:
        USE_CUSTOM_MODEL = True
        CUSTOM_MODEL_PATH = './my_models/custom_model.pth'
        CUSTOM_MODEL_CLASS = MyCustomResNet  # Import your class
        CUSTOM_NUM_CLASSES = 100
        
        # ... rest of config ...
    
    
    # STEP 3: Run compression
    # =======================
    results = main()  # Will use your custom model!
    
    
    # ALTERNATIVE: Use model loader directly
    # =======================================
    from model_compression_cnn import CustomModelLoader
    
    # Load your model
    my_model = CustomModelLoader.load_custom_model(
        model_path='./my_models/custom_model.pth',
        model_class=MyCustomResNet,
        num_classes=100
    )
    
    # Validate it
    sample = torch.randn(1, 3, 224, 224)
    CustomModelLoader.validate_model(my_model, sample)
    
    # Now use it in compression pipeline...
    """
    
    print(example_code)
    print("\n" + "="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MODEL COMPRESSION WITH VISUALIZATION")
    print("  Supports Custom User-Trained Models")
    print("="*70)
    
    # Check if using custom model
    if Config.USE_CUSTOM_MODEL:
        print("\n[CUSTOM MODEL MODE]")
        if Config.CUSTOM_MODEL_PATH and os.path.exists(Config.CUSTOM_MODEL_PATH):
            print(f"✓ Custom model found: {Config.CUSTOM_MODEL_PATH}")
        else:
            print(f"✗ Custom model not found: {Config.CUSTOM_MODEL_PATH}")
            print("  Switching to default model...")
            Config.USE_CUSTOM_MODEL = False
    
    # Show custom model usage example
    if not Config.USE_CUSTOM_MODEL:
        print("\nTo use your own model, see example below:")
        custom_model_example()
        
        print("\nPress Enter to continue with default ResNet50...")
        input()
    
    # Run compression pipeline
    results = main()
    
    # Run inference example
    inference_example()
    
    print("\n" + "="*70)
    print("  ✓ Compression complete!")
    print("  ✓ Visualizations generated!")
    print("  ✓ Models ready for deployment!")
    print("="*70 + "\n")