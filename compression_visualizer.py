"""
Model Compression Visualization & Custom Model Support
=======================================================
Comprehensive visualization toolkit and utilities for compressing
custom user-trained models (both CNN and LLM).

Features:
- Beautiful matplotlib visualizations
- Support for custom pretrained models
- Flexible architecture loading
- Interactive comparison charts
- Export-ready plots for papers/presentations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class CompressionVisualizer:
    """Comprehensive visualization toolkit for model compression results."""
    
    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_compression_comparison(
        self, 
        results: Dict[str, Dict],
        metric_name: str = "accuracy",
        title: str = "Model Compression Comparison"
    ):
        """
        Create comprehensive comparison chart for all compression stages.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            metric_name: Primary metric to display (accuracy or perplexity)
            title: Plot title
        """
        models = list(results.keys())
        
        # Extract metrics
        if metric_name == "accuracy":
            primary_metric = [results[m].get('accuracy', results[m].get('perplexity', 0)) 
                            for m in models]
            ylabel = "Accuracy (%)" if 'accuracy' in results[models[0]] else "Perplexity"
        else:
            primary_metric = [results[m].get('perplexity', results[m].get('accuracy', 0)) 
                            for m in models]
            ylabel = "Perplexity"
            
        sizes = [results[m]['model_size_mb'] for m in models]
        latencies = [results[m].get('latency_ms', 0) for m in models]
        memories = [results[m].get('memory_mb', 0) for m in models]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Model Size Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, sizes, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Model Size (MB)', fontweight='bold')
        ax1.set_title('Model Size Comparison', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Primary Metric (Accuracy/Perplexity)
        ax2 = axes[0, 1]
        colors = ['green' if i == 0 else 'orange' if i == len(models)-1 else 'steelblue' 
                  for i in range(len(models))]
        bars2 = ax2.bar(models, primary_metric, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel(ylabel, fontweight='bold')
        ax2.set_title(f'{ylabel} Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 3. Inference Latency
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, latencies, color='coral', alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Latency (ms)', fontweight='bold')
        ax3.set_title('Inference Latency Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 4. Memory Usage
        ax4 = axes[1, 1]
        bars4 = ax4.bar(models, memories, color='mediumpurple', alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Memory Usage (MB)', fontweight='bold')
        ax4.set_title('Memory Consumption Comparison', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "compression_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_accuracy_vs_size_tradeoff(
        self, 
        results: Dict[str, Dict],
        metric_name: str = "accuracy"
    ):
        """
        Create scatter plot showing accuracy vs model size tradeoff.
        
        Args:
            results: Dictionary with model metrics
            metric_name: Metric to plot (accuracy or perplexity)
        """
        models = list(results.keys())
        
        if metric_name == "accuracy":
            metric_values = [results[m].get('accuracy', 0) for m in models]
            ylabel = "Accuracy (%)"
            better_direction = "up"
        else:
            metric_values = [results[m].get('perplexity', 0) for m in models]
            ylabel = "Perplexity (lower is better)"
            better_direction = "down"
        
        sizes = [results[m]['model_size_mb'] for m in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        for i, (model, size, metric) in enumerate(zip(models, sizes, metric_values)):
            ax.scatter(size, metric, s=300, c=[colors[i]], 
                      alpha=0.7, edgecolors='black', linewidth=2, zorder=3)
            ax.annotate(model, (size, metric), 
                       textcoords="offset points", xytext=(0,10),
                       ha='center', fontsize=9, fontweight='bold')
        
        # Draw Pareto frontier (optimal tradeoff curve)
        if better_direction == "up":
            # Sort by size, then keep points with increasing accuracy
            sorted_points = sorted(zip(sizes, metric_values))
            pareto_points = []
            max_metric = -float('inf')
            for size, metric in sorted_points:
                if metric > max_metric:
                    pareto_points.append((size, metric))
                    max_metric = metric
        else:
            # For perplexity (lower is better)
            sorted_points = sorted(zip(sizes, metric_values))
            pareto_points = []
            min_metric = float('inf')
            for size, metric in sorted_points:
                if metric < min_metric:
                    pareto_points.append((size, metric))
                    min_metric = metric
        
        if len(pareto_points) > 1:
            pareto_sizes, pareto_metrics = zip(*pareto_points)
            ax.plot(pareto_sizes, pareto_metrics, 'r--', alpha=0.5, 
                   linewidth=2, label='Pareto Frontier', zorder=2)
        
        ax.set_xlabel('Model Size (MB)', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(f'{ylabel} vs Model Size Tradeoff', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "accuracy_size_tradeoff.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_compression_ratios(self, results: Dict[str, Dict]):
        """
        Create visualization of compression ratios relative to baseline.
        
        Args:
            results: Dictionary with model metrics
        """
        models = list(results.keys())
        baseline_model = models[0]  # Assume first model is baseline
        baseline_size = results[baseline_model]['model_size_mb']
        
        # Calculate compression ratios
        compression_ratios = [baseline_size / results[m]['model_size_mb'] 
                             for m in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(models))
        colors = ['green' if ratio > 1 else 'red' for ratio in compression_ratios]
        
        bars = ax.barh(y_pos, compression_ratios, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Compression Ratio (×)', fontweight='bold', fontsize=12)
        ax.set_title('Compression Ratios Relative to Baseline', 
                    fontweight='bold', fontsize=14)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        # Add value labels
        for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
            width = bar.get_width()
            label = f'{ratio:.2f}×'
            if ratio >= 1:
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                       label, ha='left', va='center', fontweight='bold')
            else:
                ax.text(width - 0.1, bar.get_y() + bar.get_height()/2.,
                       label, ha='right', va='center', fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "compression_ratios.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_speedup_analysis(self, results: Dict[str, Dict]):
        """
        Analyze and visualize inference speedup from compression.
        
        Args:
            results: Dictionary with model metrics
        """
        models = list(results.keys())
        baseline_model = models[0]
        baseline_latency = results[baseline_model].get('latency_ms', 0)
        
        if baseline_latency == 0:
            print("  ⚠ Latency data not available")
            return
        
        # Calculate speedups
        speedups = [baseline_latency / results[m]['latency_ms'] 
                   for m in models if results[m].get('latency_ms', 0) > 0]
        valid_models = [m for m in models if results[m].get('latency_ms', 0) > 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Speedup comparison
        colors = ['crimson' if s > 1 else 'steelblue' for s in speedups]
        bars = ax1.bar(valid_models, speedups, color=colors, 
                      alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Speedup (×)', fontweight='bold')
        ax1.set_title('Inference Speedup vs Baseline', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}×',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Latency breakdown
        latencies = [results[m]['latency_ms'] for m in valid_models]
        ax2.plot(valid_models, latencies, marker='o', linewidth=2, 
                markersize=10, color='darkgreen', alpha=0.7)
        ax2.set_ylabel('Latency (ms)', fontweight='bold')
        ax2.set_title('Absolute Inference Latency', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "speedup_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_summary_dashboard(self, results: Dict[str, Dict], metric_name: str = "accuracy"):
        """
        Create comprehensive dashboard with all metrics.
        
        Args:
            results: Dictionary with model metrics
            metric_name: Primary metric (accuracy or perplexity)
        """
        models = list(results.keys())
        baseline = models[0]
        
        # Extract all metrics
        if metric_name == "accuracy":
            primary_values = [results[m].get('accuracy', 0) for m in models]
            primary_label = "Accuracy (%)"
        else:
            primary_values = [results[m].get('perplexity', 0) for m in models]
            primary_label = "Perplexity"
        
        sizes = [results[m]['model_size_mb'] for m in models]
        latencies = [results[m].get('latency_ms', 0) for m in models]
        
        # Calculate improvements
        size_reduction = [(1 - s/sizes[0]) * 100 for s in sizes]
        speedup = [latencies[0] / l if l > 0 else 0 for l in latencies]
        
        # Create dashboard
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Model Compression Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Model Size Progression
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(models, sizes, marker='o', linewidth=3, markersize=12, 
                color='steelblue', alpha=0.7)
        ax1.fill_between(range(len(models)), sizes, alpha=0.3, color='steelblue')
        ax1.set_ylabel('Size (MB)', fontweight='bold')
        ax1.set_title('Model Size Progression', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Compression Summary Stats
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        summary_text = f"""
        COMPRESSION SUMMARY
        {'='*30}
        
        Baseline Size: {sizes[0]:.1f} MB
        Final Size: {sizes[-1]:.1f} MB
        
        Compression: {sizes[0]/sizes[-1]:.2f}×
        Size Reduction: {size_reduction[-1]:.1f}%
        
        {primary_label}:
          Baseline: {primary_values[0]:.2f}
          Final: {primary_values[-1]:.2f}
        
        Speedup: {speedup[-1]:.2f}×
        """
        ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        # 3. Primary Metric Evolution
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(models, primary_values, marker='s', linewidth=3, 
                markersize=12, color='green', alpha=0.7)
        ax3.set_ylabel(primary_label, fontweight='bold')
        ax3.set_title(f'{primary_label} Evolution', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Size Reduction Percentage
        ax4 = fig.add_subplot(gs[1, 2])
        colors_pie = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
        ax4.pie([sizes[0] - s for s in sizes] + [sizes[-1]], 
               labels=[f'{m}\n({sr:.0f}%)' for m, sr in zip(models[1:], size_reduction[1:])] + ['Final'],
               autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax4.set_title('Cumulative Size Reduction', fontweight='bold')
        
        # 5. Latency Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        valid_latencies = [(m, l) for m, l in zip(models, latencies) if l > 0]
        if valid_latencies:
            vm, vl = zip(*valid_latencies)
            bars = ax5.barh(vm, vl, color='coral', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Latency (ms)', fontweight='bold')
            ax5.set_title('Inference Latency', fontweight='bold')
        
        # 6. Speedup Gains
        ax6 = fig.add_subplot(gs[2, 1])
        valid_speedup = [(m, s) for m, s in zip(models, speedup) if s > 0]
        if valid_speedup:
            vm, vs = zip(*valid_speedup)
            bars = ax6.bar(vm, vs, color='crimson', alpha=0.7, edgecolor='black')
            ax6.set_ylabel('Speedup (×)', fontweight='bold')
            ax6.set_title('Inference Speedup', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            ax6.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 7. Overall Efficiency Score
        ax7 = fig.add_subplot(gs[2, 2])
        # Calculate efficiency: higher primary metric, lower size
        if metric_name == "accuracy":
            efficiency = [p / (s/1000) for p, s in zip(primary_values, sizes)]
            efficiency_label = "Accuracy per MB"
        else:
            efficiency = [(100/p) / (s/1000) for p, s in zip(primary_values, sizes)]
            efficiency_label = "Efficiency Score"
        
        norm_efficiency = [e/max(efficiency)*100 for e in efficiency]
        ax7.barh(models, norm_efficiency, color='mediumpurple', 
                alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Normalized Score', fontweight='bold')
        ax7.set_title('Overall Efficiency', fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "compression_dashboard.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def generate_all_plots(self, results: Dict[str, Dict], metric_name: str = "accuracy"):
        """
        Generate all visualization plots.
        
        Args:
            results: Dictionary with model metrics
            metric_name: Primary metric to use
        """
        print("\n[INFO] Generating visualization plots...")
        print(f"  Metric: {metric_name}")
        print(f"  Models: {list(results.keys())}")
        
        self.plot_compression_comparison(results, metric_name)
        self.plot_accuracy_vs_size_tradeoff(results, metric_name)
        self.plot_compression_ratios(results)
        self.plot_speedup_analysis(results)
        self.plot_summary_dashboard(results, metric_name)
        
        print(f"\n✓ All visualizations saved to: {self.save_dir}")


# ============================================================================
# CUSTOM MODEL LOADING UTILITIES
# ============================================================================

class CustomModelLoader:
    """
    Utilities for loading custom user-trained models for compression.
    """
    
    @staticmethod
    def load_custom_cnn(
        model_path: str,
        model_class = None,
        num_classes: int = 10,
        device: str = "cuda"
    ):
        """
        Load custom CNN model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            model_class: Model architecture class (if None, tries to infer)
            num_classes: Number of output classes
            device: Device to load model on
        
        Returns:
            Loaded model
        
        Example:
            # Load your custom trained model
            model = CustomModelLoader.load_custom_cnn(
                model_path="./my_models/custom_resnet.pth",
                model_class=MyCustomResNet,
                num_classes=100
            )
        """
        import torch
        import torch.nn as nn
        
        print(f"\n[INFO] Loading custom CNN from: {model_path}")
        
        # Load checkpoint
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
            state_dict = checkpoint
        
        # If model class provided, instantiate it
        if model_class is not None:
            print(f"  Using provided model class: {model_class.__name__}")
            model = model_class(num_classes=num_classes)
            model.load_state_dict(state_dict)
        else:
            # Try to create a generic model wrapper
            print("  Creating generic model wrapper...")
            
            class GenericModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.state_dict_keys = list(state_dict.keys())
                
                def forward(self, x):
                    raise NotImplementedError(
                        "Model architecture not specified. "
                        "Please provide model_class parameter."
                    )
            
            model = GenericModel()
            model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        
        return model
    
    @staticmethod
    def load_custom_llm(
        model_path: str,
        tokenizer_path: Optional[str] = None,
        model_class = None,
        trust_remote_code: bool = False
    ):
        """
        Load custom LLM/Transformer model.
        
        Args:
            model_path: Path to model directory or checkpoint
            tokenizer_path: Path to tokenizer (if different from model_path)
            model_class: Model class (if not using Hugging Face format)
            trust_remote_code: Whether to trust remote code
        
        Returns:
            Tuple of (model, tokenizer)
        
        Example:
            # Load your fine-tuned GPT model
            model, tokenizer = CustomModelLoader.load_custom_llm(
                model_path="./my_models/finetuned_gpt",
                tokenizer_path="./my_models/finetuned_gpt"
            )
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\n[INFO] Loading custom LLM from: {model_path}")
        
        # Load tokenizer
        tok_path = tokenizer_path if tokenizer_path else model_path
        print(f"  Loading tokenizer from: {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print(f"  Loading model...")
        if model_class is not None:
            model = model_class.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        
        return model, tokenizer
    
    @staticmethod
    def validate_model_for_compression(model, sample_input):
        """
        Validate that model is compatible with compression pipeline.
        
        Args:
            model: Model to validate
            sample_input: Sample input tensor
        
        Returns:
            True if valid, raises exception otherwise
        """
        import torch
        
        print("\n[INFO] Validating model for compression...")
        
        try:
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
            print("  ✓ Forward pass successful")
            
            # Check if model has state_dict
            state_dict = model.state_dict()
            print(f"  ✓ State dict accessible ({len(state_dict)} parameters)")
            
            # Check if model can be saved
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                torch.save(model.state_dict(), tmp.name)
            print("  ✓ Model serialization successful")
            
            print("\n✓ Model validation passed! Ready for compression.")
            return True
            
        except Exception as e:
            print(f"\n✗ Model validation failed: {e}")
            raise


# ============================================================================
# EXAMPLE USAGE FOR CUSTOM MODELS
# ============================================================================

def compress_custom_cnn_example():
    """
    Example: Compress your own custom CNN model.
    """
    print("\n" + "="*70)
    print("  EXAMPLE: Compressing Custom CNN Model")
    print("="*70)
    
    print("""
    To compress your own CNN model:
    
    1. Train your model normally:
       ```python
       my_model = MyCustomResNet(num_classes=100)
       # ... training code ...
       torch.save(my_model.state_dict(), 'my_model.pth')
       ```
    
    2. Load it for compression:
       ```python
       from compression_utils import CustomModelLoader
       
       # Load your model
       teacher = CustomModelLoader.load_custom_cnn(
           model_path='./my_models/my_model.pth',
           model_class=MyCustomResNet,
           num_classes=100
       )
       
       # Validate it
       sample_input = torch.randn(1, 3, 224, 224)
       CustomModelLoader.validate_model_for_compression(teacher, sample_input)
       ```
    
    3. Run compression pipeline:
       ```python
       # In the CNN compression script, replace:
       # teacher = prepare_teacher_model()
       
       # With your custom model:
       teacher = CustomModelLoader.load_custom_cnn(
           model_path='./my_models/my_model.pth',
           model_class=MyCustomResNet,
           num_classes=100
       )
       
       # Then continue with normal pipeline...
       # Stage 1: QAT
       # Stage 2: Mixed-precision
       # Stage 3: Distillation
       # etc.
       ```
    
    4. Visualize results:
       ```python
       visualizer = CompressionVisualizer('./my_visualizations')
       visualizer.generate_all_plots(results, metric_name='accuracy')
       ```
    """)
    
    print("\nSee full working example below:")
    print("-" * 70)


def compress_custom_llm_example():
    """
    Example: Compress your own fine-tuned LLM.
    """
    print("\n" + "="*70)
    print("  EXAMPLE: Compressing Custom LLM Model")
    print("="*70)
    
    print("""
    To compress your own LLM:
    
    1. Fine-tune your model:
       ```python
       from transformers import AutoModelForCausalLM, Trainer
       
       model = AutoModelForCausalLM.from_pretrained('gpt2')
       # ... fine-tuning code ...
       model.save_pretrained('./my_models/my_finetuned_gpt')
       ```
    
    2. Load it for compression:
       ```python
       from compression_utils import CustomModelLoader
       
       # Load your model
       teacher, tokenizer = CustomModelLoader.load_custom_llm(
           model_path='./my_models/my_finetuned_gpt',
           tokenizer_path='./my_models/my_finetuned_gpt'
       )
       
       # Validate it
       sample_input = tokenizer("Hello world", return_tensors="pt")
       CustomModelLoader.validate_model_for_compression(
           teacher, sample_input['input_ids']
       )
       ```
    
    3. Run compression with your model:
       ```python
       # In LLM compression script, replace:
       # teacher, teacher_tokenizer = load_teacher_model()
       
       # With your custom model:
       teacher, tokenizer = CustomModelLoader.load_custom_llm(
           model_path='./my_models/my_finetuned_gpt'
       )
       
       # Continue with compression stages:
       # - 4-bit quantization
       # - QLoRA fine-tuning
       # - Knowledge distillation
       # etc.
       ```
    
    4. Visualize compression results:
       ```python
       visualizer = CompressionVisualizer('./my_visualizations')
       visualizer.generate_all_plots(results, metric_name='perplexity')
       ```
    """)


# ============================================================================
# COMPLETE WORKING EXAMPLE WITH VISUALIZATION
# ============================================================================

def complete_compression_example_with_viz():
    """
    Complete example showing compression pipeline with visualization.
    """
    print("\n" + "="*70)
    print("  COMPLETE EXAMPLE: Compression + Visualization")
    print("="*70)
    
    # Example results (would come from actual compression pipeline)
    example_results = {
        "teacher_fp32": {
            "accuracy": 94.5,
            "model_size_mb": 90.2,
            "parameters": 23500000,
            "trainable_parameters": 23500000,
            "memory_mb": 450.0,
            "latency_ms": 45.2
        },
        "qat_int8": {
            "accuracy": 93.8,
            "model_size_mb": 23.1,
            "parameters": 23500000,
            "trainable_parameters": 23500000,
            "memory_mb": 120.0,
            "latency_ms": 22.5
        },
        "mixed_precision": {
            "accuracy": 94.1,
            "model_size_mb": 25.4,
            "parameters": 23500000,
            "trainable_parameters": 23500000,
            "memory_mb": 135.0,
            "latency_ms": 24.8
        },
        "distilled": {
            "accuracy": 92.7,
            "model_size_mb": 44.5,
            "parameters": 11000000,
            "trainable_parameters": 11000000,
            "memory_mb": 220.0,
            "latency_ms": 18.3
        },
        "final_compressed": {
            "accuracy": 91.9,
            "model_size_mb": 11.8,
            "parameters": 11000000,
            "trainable_parameters": 11000000,
            "memory_mb": 65.0,
            "latency_ms": 9.1
        }
    }
    
    # Create visualizer
    visualizer = CompressionVisualizer(save_dir="./example_visualizations")
    
    # Generate all plots
    visualizer.generate_all_plots(example_results, metric_name="accuracy")
    
    print("\n✓ Example visualizations generated!")
    print("  Check ./example_visualizations/ directory")


# ============================================================================
# INTEGRATION WITH EXISTING SCRIPTS
# ============================================================================

def integrate_with_cnn_script():
    """
    Instructions for integrating visualization with CNN compression script.
    """
    print("""
    ========================================
    INTEGRATION WITH CNN COMPRESSION SCRIPT
    ========================================
    
    Add these lines to your model_compression_cnn.py:
    
    1. At the top, import the visualizer:
       ```python
       from compression_utils import CompressionVisualizer
       ```
    
    2. After the main() function completes and returns results:
       ```python
       def main():
           # ... existing compression pipeline ...
           
           # Add visualization at the end:
           visualizer = CompressionVisualizer(
               save_dir=os.path.join(Config.SAVE_DIR, "visualizations")
           )
           visualizer.generate_all_plots(results, metric_name='accuracy')
           
           return results
       ```
    
    3. To use custom models, modify prepare_teacher_model():
       ```python
       def prepare_teacher_model(
           custom_path: Optional[str] = None,
           custom_class = None
       ):
           if custom_path:
               return CustomModelLoader.load_custom_cnn(
                   model_path=custom_path,
                   model_class=custom_class,
                   num_classes=10  # or your number of classes
               )
           else:
               # Use default ResNet50
               model = models.resnet50(pretrained=True)
               # ... rest of original code ...
       ```
    """)


def integrate_with_llm_script():
    """
    Instructions for integrating visualization with LLM compression script.
    """
    print("""
    ========================================
    INTEGRATION WITH LLM COMPRESSION SCRIPT
    ========================================
    
    Add these lines to your model_compression_llm.py:
    
    1. Import the visualizer:
       ```python
       from compression_utils import CompressionVisualizer
       ```
    
    2. Add visualization after main() completes:
       ```python
       def main():
           # ... existing compression pipeline ...
           
           # Add visualization:
           visualizer = CompressionVisualizer(
               save_dir=os.path.join(Config.SAVE_DIR, "visualizations")
           )
           visualizer.generate_all_plots(results, metric_name='perplexity')
           
           return results
       ```
    
    3. To use custom LLM models:
       ```python
       def load_teacher_model(
           quantize: bool = False,
           custom_path: Optional[str] = None
       ):
           if custom_path:
               model, tokenizer = CustomModelLoader.load_custom_llm(
                   model_path=custom_path,
                   tokenizer_path=custom_path
               )
               return model, tokenizer
           else:
               # Use default GPT-2
               # ... rest of original code ...
       ```
    """)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MODEL COMPRESSION VISUALIZATION & CUSTOM MODEL TOOLKIT")
    print("="*70)
    
    # Show examples
    compress_custom_cnn_example()
    compress_custom_llm_example()
    
    # Generate example visualizations
    complete_compression_example_with_viz()
    
    # Show integration instructions
    print("\n" + "="*70)
    print("  INTEGRATION INSTRUCTIONS")
    print("="*70)
    integrate_with_cnn_script()
    integrate_with_llm_script()
    
    print("\n" + "="*70)
    print("  Quick Start for Custom Models:")
    print("="*70)
    print("""
    # CNN Example:
    from compression_utils import CustomModelLoader, CompressionVisualizer
    
    # Load your model
    model = CustomModelLoader.load_custom_cnn(
        model_path='path/to/your/model.pth',
        model_class=YourModelClass,
        num_classes=100
    )
    
    # Run compression pipeline (modify the scripts)
    # results = compress_pipeline(model)
    
    # Visualize results
    viz = CompressionVisualizer('./visualizations')
    viz.generate_all_plots(results, 'accuracy')
    
    # LLM Example:
    model, tokenizer = CustomModelLoader.load_custom_llm(
        model_path='path/to/your/llm'
    )
    
    # Run LLM compression pipeline
    # results = compress_llm_pipeline(model, tokenizer)
    
    # Visualize
    viz = CompressionVisualizer('./visualizations')
    viz.generate_all_plots(results, 'perplexity')
    """)
    
    print("\n✓ Toolkit ready! See examples above.")
    print("="*70 + "\n")