"""
Research Analysis Utilities for TianWen Framework.

This module provides utilities for analyzing models, features, and results
to facilitate scientific research and debugging.

Features:
- Model analysis (parameters, FLOPs, memory)
- Feature visualization
- Attention map visualization
- Ablation study support
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Container for model statistics."""
    total_params: int
    trainable_params: int
    frozen_params: int
    total_memory_mb: float
    param_groups: Dict[str, int]

    def __repr__(self) -> str:
        return (
            f"ModelStats(\n"
            f"  total_params={self.total_params:,}\n"
            f"  trainable_params={self.trainable_params:,}\n"
            f"  frozen_params={self.frozen_params:,}\n"
            f"  memory={self.total_memory_mb:.2f} MB\n"
            f")"
        )


class ModelAnalyzer:
    """
    Utility for analyzing model architecture and performance.

    Provides:
    - Parameter counting by layer/module
    - Memory estimation
    - Forward pass profiling
    - Gradient flow analysis

    Example:
        >>> analyzer = ModelAnalyzer(model)
        >>> stats = analyzer.get_stats()
        >>> print(analyzer.summary())
        >>> analyzer.profile_forward(sample_input)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize model analyzer.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def get_stats(self) -> ModelStats:
        """
        Get model statistics.

        Returns:
            ModelStats object
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        param_groups = {}

        for name, param in self.model.named_parameters():
            count = param.numel()
            total_params += count

            if param.requires_grad:
                trainable_params += count
            else:
                frozen_params += count

            # Group by top-level module
            group = name.split(".")[0]
            param_groups[group] = param_groups.get(group, 0) + count

        # Estimate memory
        total_memory = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        total_memory_mb = total_memory / (1024 * 1024)

        return ModelStats(
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            total_memory_mb=total_memory_mb,
            param_groups=param_groups,
        )

    def summary(
        self,
        input_size: Optional[Tuple[int, ...]] = None,
        depth: int = 3,
    ) -> str:
        """
        Generate model summary.

        Args:
            input_size: Input tensor size for shape inference
            depth: Maximum depth of modules to show

        Returns:
            Summary string
        """
        lines = ["=" * 80]
        lines.append(f"Model: {self.model.__class__.__name__}")
        lines.append("=" * 80)

        stats = self.get_stats()
        lines.append(f"Total parameters: {stats.total_params:,}")
        lines.append(f"Trainable parameters: {stats.trainable_params:,}")
        lines.append(f"Frozen parameters: {stats.frozen_params:,}")
        lines.append(f"Estimated memory: {stats.total_memory_mb:.2f} MB")
        lines.append("-" * 80)

        lines.append("\nParameter groups:")
        for group, count in sorted(stats.param_groups.items(), key=lambda x: -x[1]):
            pct = count / stats.total_params * 100
            lines.append(f"  {group}: {count:,} ({pct:.1f}%)")

        lines.append("-" * 80)

        # Module hierarchy
        lines.append("\nModule hierarchy:")
        self._add_module_summary(lines, self.model, depth=depth)

        lines.append("=" * 80)

        return "\n".join(lines)

    def _add_module_summary(
        self,
        lines: List[str],
        module: nn.Module,
        prefix: str = "",
        depth: int = 3,
        current_depth: int = 0,
    ) -> None:
        """Add module summary lines recursively."""
        if current_depth >= depth:
            return

        for name, child in module.named_children():
            # Count parameters in this module
            num_params = sum(p.numel() for p in child.parameters(recurse=False))
            total_params = sum(p.numel() for p in child.parameters())

            indent = "  " * current_depth
            class_name = child.__class__.__name__

            if num_params > 0 or total_params > 0:
                lines.append(
                    f"{indent}{prefix}{name} ({class_name}): "
                    f"{total_params:,} params"
                )

            self._add_module_summary(
                lines, child,
                prefix=f"{name}.",
                depth=depth,
                current_depth=current_depth + 1,
            )

    def profile_forward(
        self,
        input_tensor: Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Profile forward pass timing.

        Args:
            input_tensor: Input tensor for forward pass
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        import time

        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(warmup_runs):
                _ = self.model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Profile
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append(end - start)

        import numpy as np
        times = np.array(times)

        return {
            "mean_ms": times.mean() * 1000,
            "std_ms": times.std() * 1000,
            "min_ms": times.min() * 1000,
            "max_ms": times.max() * 1000,
            "fps": 1.0 / times.mean(),
        }

    def register_activation_hooks(
        self,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        """
        Register hooks to capture layer activations.

        Args:
            layer_names: Specific layers to capture (None for all)
        """
        self._clear_hooks()
        self._activations.clear()

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self._activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)

    def register_gradient_hooks(
        self,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        """
        Register hooks to capture gradients.

        Args:
            layer_names: Specific layers to capture (None for all)
        """
        self._clear_hooks()
        self._gradients.clear()

        def make_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self._gradients[name] = grad_output[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_full_backward_hook(make_hook(name))
                self._hooks.append(hook)

    def get_activations(self) -> Dict[str, Tensor]:
        """Get captured activations."""
        return dict(self._activations)

    def get_gradients(self) -> Dict[str, Tensor]:
        """Get captured gradients."""
        return dict(self._gradients)

    def _clear_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def analyze_gradient_flow(
        self,
        loss: Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze gradient flow through the model.

        Args:
            loss: Loss tensor to backpropagate

        Returns:
            Dictionary mapping layer name to gradient statistics
        """
        loss.backward()

        gradient_stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                gradient_stats[name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.max().item(),
                    "min": grad.min().item(),
                    "norm": grad.norm().item(),
                }

        return gradient_stats

    def __del__(self):
        """Clean up hooks on deletion."""
        self._clear_hooks()


class FeatureVisualizer:
    """
    Utility for visualizing intermediate features and attention maps.

    Example:
        >>> visualizer = FeatureVisualizer(model)
        >>> visualizer.visualize_features(image, layer="backbone.layer4")
        >>> visualizer.visualize_attention(image)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        """
        Initialize feature visualizer.

        Args:
            model: PyTorch model
            device: Device for computation
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks = []
        self._features = {}

    def visualize_features(
        self,
        image: Tensor,
        layer_name: str,
        output_path: Optional[str] = None,
        num_channels: int = 16,
        figsize: Tuple[int, int] = (16, 8),
    ) -> Optional[Tensor]:
        """
        Visualize feature maps from a specific layer.

        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            layer_name: Name of layer to visualize
            output_path: Optional path to save visualization
            num_channels: Number of channels to visualize
            figsize: Figure size for visualization

        Returns:
            Feature tensor if no output_path specified
        """
        # Register hook
        self._features.clear()

        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self._features[layer_name] = output.detach()

        # Find and hook the layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer not found: {layer_name}")

        handle = target_module.register_forward_hook(hook)

        try:
            # Forward pass
            if image.dim() == 3:
                image = image.unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                image = image.to(self.device)
                _ = self.model(image)

            # Get features
            features = self._features[layer_name]

            if output_path:
                self._plot_features(features, output_path, num_channels, figsize)
            else:
                return features

        finally:
            handle.remove()

    def _plot_features(
        self,
        features: Tensor,
        output_path: str,
        num_channels: int,
        figsize: Tuple[int, int],
    ) -> None:
        """Plot feature maps."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for visualization")
            return

        # Get first batch item
        features = features[0].cpu()
        num_channels = min(num_channels, features.shape[0])

        # Create grid
        rows = int(num_channels ** 0.5)
        cols = (num_channels + rows - 1) // rows

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_channels > 1 else [axes]

        for i in range(num_channels):
            ax = axes[i]
            feat = features[i].numpy()
            ax.imshow(feat, cmap="viridis")
            ax.set_title(f"Channel {i}")
            ax.axis("off")

        # Hide empty axes
        for i in range(num_channels, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature visualization to: {output_path}")

    def visualize_attention(
        self,
        image: Tensor,
        layer_name: str,
        output_path: Optional[str] = None,
        head_idx: int = 0,
    ) -> Optional[Tensor]:
        """
        Visualize attention maps from transformer layers.

        Args:
            image: Input image tensor
            layer_name: Attention layer name
            output_path: Optional path to save visualization
            head_idx: Attention head index to visualize

        Returns:
            Attention tensor if no output_path specified
        """
        self._features.clear()

        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Many attention layers return (output, attention_weights)
                self._features["attention"] = output[1].detach()
            else:
                self._features["attention"] = output.detach()

        # Find attention layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer not found: {layer_name}")

        handle = target_module.register_forward_hook(hook)

        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                image = image.to(self.device)
                _ = self.model(image)

            attention = self._features.get("attention")

            if attention is None:
                logger.warning(f"No attention captured from {layer_name}")
                return None

            if output_path:
                self._plot_attention(attention, output_path, head_idx)
            else:
                return attention

        finally:
            handle.remove()

    def _plot_attention(
        self,
        attention: Tensor,
        output_path: str,
        head_idx: int,
    ) -> None:
        """Plot attention map."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for visualization")
            return

        # Shape: [batch, heads, seq_len, seq_len] or similar
        attn = attention[0].cpu()  # First batch item

        if attn.dim() == 3:
            attn = attn[head_idx]  # Select head
        elif attn.dim() == 2:
            pass  # Already 2D

        plt.figure(figsize=(8, 8))
        plt.imshow(attn.numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Attention Map (Head {head_idx})")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved attention visualization to: {output_path}")

    def create_saliency_map(
        self,
        image: Tensor,
        target_class: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Tensor:
        """
        Create saliency map showing input importance.

        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            target_class: Class index for gradient (None for max)
            output_path: Optional path to save visualization

        Returns:
            Saliency map tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device).requires_grad_(True)

        self.model.eval()
        output = self.model(image)

        # Handle different output formats
        if hasattr(output, "detection_output"):
            # TianWen fusion output
            scores = output.detection_output.outputs[0].scores
            if len(scores) > 0:
                score = scores.max()
            else:
                logger.warning("No detections for saliency map")
                return torch.zeros_like(image)
        elif isinstance(output, dict) and "scores" in output:
            score = output["scores"].max()
        elif isinstance(output, Tensor):
            if target_class is not None:
                score = output[0, target_class]
            else:
                score = output.max()
        else:
            raise ValueError(f"Unsupported output format: {type(output)}")

        # Backward pass
        score.backward()

        # Get saliency
        saliency = image.grad.abs()
        saliency = saliency.max(dim=1, keepdim=True)[0]  # Max over channels

        if output_path:
            self._plot_saliency(image, saliency, output_path)

        return saliency.squeeze()

    def _plot_saliency(
        self,
        image: Tensor,
        saliency: Tensor,
        output_path: str,
    ) -> None:
        """Plot saliency map overlaid on image."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for visualization")
            return

        img = image[0].detach().cpu().permute(1, 2, 0).numpy()
        sal = saliency[0, 0].detach().cpu().numpy()

        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(sal, cmap="hot")
        axes[1].set_title("Saliency Map")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(sal, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved saliency map to: {output_path}")


class AblationStudy:
    """
    Utility for conducting systematic ablation studies.

    Automates the process of disabling components and measuring
    their impact on model performance.

    Example:
        >>> ablation = AblationStudy(model, eval_fn)
        >>> ablation.add_component("distillation", disable_fn, enable_fn)
        >>> ablation.add_component("feature_fusion", disable_fn2, enable_fn2)
        >>> results = ablation.run()
    """

    def __init__(
        self,
        model: nn.Module,
        eval_fn: Callable[[nn.Module], Dict[str, float]],
        base_name: str = "full_model",
    ):
        """
        Initialize ablation study.

        Args:
            model: Model to study
            eval_fn: Function that evaluates model and returns metrics dict
            base_name: Name for full model baseline
        """
        self.model = model
        self.eval_fn = eval_fn
        self.base_name = base_name

        self._components: Dict[str, Dict[str, Callable]] = OrderedDict()
        self._results: Dict[str, Dict[str, float]] = {}

    def add_component(
        self,
        name: str,
        disable_fn: Callable[[nn.Module], None],
        enable_fn: Callable[[nn.Module], None],
    ) -> None:
        """
        Add a component for ablation.

        Args:
            name: Component name
            disable_fn: Function to disable the component
            enable_fn: Function to re-enable the component
        """
        self._components[name] = {
            "disable": disable_fn,
            "enable": enable_fn,
        }

    def run(
        self,
        include_combinations: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study.

        Args:
            include_combinations: Include combinations of disabled components

        Returns:
            Dictionary mapping configuration name to metrics
        """
        # Evaluate full model
        logger.info(f"Evaluating: {self.base_name}")
        self._results[self.base_name] = self.eval_fn(self.model)

        # Ablate each component individually
        for comp_name, funcs in self._components.items():
            config_name = f"without_{comp_name}"
            logger.info(f"Evaluating: {config_name}")

            # Disable component
            funcs["disable"](self.model)

            # Evaluate
            self._results[config_name] = self.eval_fn(self.model)

            # Re-enable
            funcs["enable"](self.model)

        # Optionally test combinations
        if include_combinations and len(self._components) > 1:
            from itertools import combinations

            for r in range(2, len(self._components) + 1):
                for combo in combinations(self._components.keys(), r):
                    config_name = "without_" + "_and_".join(combo)
                    logger.info(f"Evaluating: {config_name}")

                    # Disable all in combo
                    for comp in combo:
                        self._components[comp]["disable"](self.model)

                    # Evaluate
                    self._results[config_name] = self.eval_fn(self.model)

                    # Re-enable all
                    for comp in combo:
                        self._components[comp]["enable"](self.model)

        return self._results

    def print_results(
        self,
        metric: Optional[str] = None,
    ) -> None:
        """
        Print ablation study results.

        Args:
            metric: Specific metric to highlight (shows impact)
        """
        if not self._results:
            print("No results. Run the ablation study first.")
            return

        # Get all metrics
        all_metrics = set()
        for results in self._results.values():
            all_metrics.update(results.keys())
        all_metrics = sorted(all_metrics)

        # Print header
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)

        header = f"{'Configuration':<35} | " + " | ".join(f"{m:>12}" for m in all_metrics)
        print(header)
        print("-" * len(header))

        # Print rows
        baseline = self._results.get(self.base_name, {})

        for config, results in self._results.items():
            values = []
            for m in all_metrics:
                val = results.get(m, 0)
                base_val = baseline.get(m, val)

                if config == self.base_name or base_val == 0:
                    values.append(f"{val:>12.4f}")
                else:
                    diff = (val - base_val) / base_val * 100
                    sign = "+" if diff > 0 else ""
                    values.append(f"{val:>8.4f} ({sign}{diff:>+5.1f}%)")

            print(f"{config:<35} | " + " | ".join(values))

        print("=" * 80)

        # Component importance
        if metric and metric in all_metrics:
            print(f"\nComponent Importance (by {metric}):")
            base_val = baseline.get(metric, 0)

            impacts = []
            for comp in self._components:
                config = f"without_{comp}"
                if config in self._results:
                    val = self._results[config].get(metric, 0)
                    impact = (base_val - val) / base_val * 100 if base_val != 0 else 0
                    impacts.append((comp, impact))

            # Sort by impact
            impacts.sort(key=lambda x: -abs(x[1]))

            for comp, impact in impacts:
                bar = "█" * int(abs(impact) / 2)
                sign = "+" if impact > 0 else "-"
                print(f"  {comp:<25}: {sign}{abs(impact):.2f}% {bar}")

    def save_results(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save ablation results to file.

        Args:
            output_path: Output file path
        """
        import json

        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(self._results, f, indent=2)

        logger.info(f"Saved ablation results to: {output_path}")


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = "cpu",
) -> int:
    """
    Estimate FLOPs for a model.

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W) or (B, C, H, W)
        device: Device for computation

    Returns:
        Estimated FLOPs count
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logger.warning("fvcore is required for FLOP counting: pip install fvcore")
        return 0

    if len(input_size) == 3:
        input_size = (1,) + input_size

    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(input_size, device=device)

    flops = FlopCountAnalysis(model, input_tensor)

    return flops.total()


def print_layer_shapes(
    model: nn.Module,
    input_tensor: Tensor,
) -> None:
    """
    Print shapes of all intermediate layers.

    Args:
        model: PyTorch model
        input_tensor: Input tensor for forward pass
    """
    hooks = []
    shapes = OrderedDict()

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, Tensor):
                shapes[name] = tuple(output.shape)
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)

        print("\nLayer Output Shapes:")
        print("-" * 60)
        for name, shape in shapes.items():
            print(f"{name:<40} {str(shape):>20}")

    finally:
        for hook in hooks:
            hook.remove()
