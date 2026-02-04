"""Compute pairwise synergy/interaction between MNIST components.

For logit y_d(x) and components i, j:
  Δ_i = y_d(x) - y_d(x with component i ablated)
  Δ_{i,j} = y_d(x) - y_d(x with both i and j ablated)

Synergy/Interaction:
  I_{i,j} = Δ_{i,j} - Δ_i - Δ_j

I_{i,j} ≈ 0: mostly additive
I_{i,j} ≠ 0: genuine nonlinear interaction
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo
from spd.utils.module_utils import expand_module_patterns


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_component_model(
    experiment_dir: str | Path,
    checkpoint_step: int | None = None,
    device: str = "cpu",
) -> tuple[ComponentModel, TwoLayerMLP, Config]:
    """Load a trained component model from experiment directory."""
    experiment_path = Path(experiment_dir)

    # Load config
    config_path = experiment_path / "final_config.yaml"
    with open(config_path) as f:
        import yaml

        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Load the original model
    model = TwoLayerMLP(input_size=784, hidden_size=128, num_classes=10)
    model_path = experiment_path / "trained_mlp.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Create component model
    module_path_info = expand_module_patterns(model, config.all_module_info)
    component_model = ComponentModel(
        target_model=model,
        module_path_info=module_path_info,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )

    # Find and load checkpoint
    if checkpoint_step is None:
        checkpoints = list(experiment_path.glob("model_*.pth"))
        assert checkpoints, f"No checkpoints found in {experiment_path}"
        checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    else:
        checkpoint_path = experiment_path / f"model_{checkpoint_step}.pth"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
    component_model.load_state_dict(checkpoint_weights)
    component_model.to(device)
    component_model.eval()

    return component_model, model, config


def get_alive_components(
    component_model: ComponentModel,
    test_images: torch.Tensor,
    layer: str = "fc1",
    threshold: float = 0.01,
    n_samples: int = 1000,
) -> list[int]:
    """Get list of alive component indices based on causal importance."""
    device = next(component_model.parameters()).device
    n_components = component_model.components[layer].V.shape[1]

    # Compute CI for samples
    ci_values = []
    with torch.no_grad():
        for i in range(min(n_samples, len(test_images))):
            image_flat = test_images[i : i + 1].to(device)

            # Get pre-weight activations
            pre_weight_acts = {}
            if "fc1" in component_model.components:
                pre_weight_acts["fc1"] = image_flat
            if "fc2" in component_model.components:
                # We know target_model is TwoLayerMLP with fc1 attribute
                target = cast(TwoLayerMLP, component_model.target_model)
                hidden = torch.relu(target.fc1(image_flat))
                pre_weight_acts["fc2"] = hidden

            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sampling="continuous",
                detach_inputs=False,
            )

            if layer in ci_outputs.lower_leaky:
                ci_values.append(ci_outputs.lower_leaky[layer][0].cpu().numpy())

    ci_values = np.array(ci_values)

    # Component is alive if >0.5% of samples have CI > threshold
    alive = []
    for comp_idx in range(n_components):
        ci = ci_values[:, comp_idx]
        if (ci > threshold).mean() > 0.005:
            alive.append(comp_idx)

    return alive


def compute_logits_with_ablation(
    component_model: ComponentModel,
    images: torch.Tensor,
    ablated_components: list[int],
    layer: str = "fc1",
) -> torch.Tensor:
    """Compute model logits with specified components ablated.

    Args:
        component_model: The SPD component model
        images: Input images (batch, 784)
        ablated_components: List of component indices to ablate (set mask to 0)
        layer: Which layer to ablate components from

    Returns:
        Logits tensor (batch, 10)
    """
    device = next(component_model.parameters()).device
    images = images.to(device)
    n_components = component_model.components[layer].V.shape[1]
    batch_size = images.shape[0]

    # Create component mask: 1 for active, 0 for ablated
    component_mask = torch.ones(batch_size, n_components, device=device)
    for comp_idx in ablated_components:
        component_mask[:, comp_idx] = 0.0

    # Create mask_infos for all layers
    mask_infos = {}
    for layer_name in component_model.components:
        n_comp = component_model.components[layer_name].V.shape[1]
        if layer_name == layer:
            mask = component_mask
        else:
            # Keep all components active for other layers
            mask = torch.ones(batch_size, n_comp, device=device)

        mask_infos[layer_name] = ComponentsMaskInfo(
            component_mask=mask,
            routing_mask="all",
        )

    # Forward pass with ablation
    with torch.no_grad():
        output = component_model(images, mask_infos=mask_infos)

    return output


def compute_synergy_matrix(
    component_model: ComponentModel,
    images: torch.Tensor,
    labels: np.ndarray,
    alive_components: list[int],
    digit_class: int,
    layer: str = "fc1",
) -> np.ndarray:
    """Compute pairwise synergy matrix for a specific digit class.

    Args:
        component_model: The SPD component model
        images: Test images (n_samples, 784)
        labels: Test labels (n_samples,)
        alive_components: List of alive component indices
        digit_class: Which digit class to compute synergy for
        layer: Which layer to analyze

    Returns:
        Synergy matrix of shape (n_alive, n_alive)
    """
    n_alive = len(alive_components)
    device = next(component_model.parameters()).device

    # Filter to samples of this digit class
    class_mask = labels == digit_class
    class_images = images[class_mask]

    if len(class_images) == 0:
        return np.zeros((n_alive, n_alive))

    # Limit to reasonable batch size
    max_samples = min(100, len(class_images))
    class_images = class_images[:max_samples].to(device)

    # Compute baseline logits (no ablation)
    baseline_logits = compute_logits_with_ablation(
        component_model, class_images, ablated_components=[], layer=layer
    )
    y_full = baseline_logits[:, digit_class].cpu().numpy()  # (n_samples,)

    # Compute single-component ablation effects: Δ_i = y(x) - y(x with i ablated)
    delta_single = np.zeros((len(class_images), n_alive))
    for i, comp_i in enumerate(
        tqdm(alive_components, desc=f"Single ablations (digit {digit_class})")
    ):
        ablated_logits = compute_logits_with_ablation(
            component_model, class_images, ablated_components=[comp_i], layer=layer
        )
        y_ablated_i = ablated_logits[:, digit_class].cpu().numpy()
        delta_single[:, i] = y_full - y_ablated_i

    # Compute pairwise ablation effects and synergy
    synergy_matrix = np.zeros((n_alive, n_alive))
    for i, comp_i in enumerate(
        tqdm(alive_components, desc=f"Pairwise ablations (digit {digit_class})")
    ):
        for j, comp_j in enumerate(alive_components):
            if i == j:
                # Self-interaction is 0 by definition
                synergy_matrix[i, j] = 0.0
                continue

            # Only compute upper triangle (symmetric)
            if j < i:
                synergy_matrix[i, j] = synergy_matrix[j, i]
                continue

            # Compute Δ_{i,j} = y(x) - y(x with i,j ablated)
            ablated_logits = compute_logits_with_ablation(
                component_model, class_images, ablated_components=[comp_i, comp_j], layer=layer
            )
            y_ablated_ij = ablated_logits[:, digit_class].cpu().numpy()
            delta_ij = y_full - y_ablated_ij

            # Synergy: I_{i,j} = Δ_{i,j} - Δ_i - Δ_j
            # Average across samples
            synergy = (delta_ij - delta_single[:, i] - delta_single[:, j]).mean()
            synergy_matrix[i, j] = synergy
            synergy_matrix[j, i] = synergy  # Symmetric

    return synergy_matrix


def plot_synergy_heatmaps(
    synergy_matrices: dict[int, np.ndarray],
    alive_components: list[int],
    output_dir: Path,
    layer: str = "fc1",
):
    """Plot synergy heatmaps for all digit classes."""
    n_alive = len(alive_components)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    # Use symmetric scale centered at 0 (full range)
    all_values = np.concatenate([m.flatten() for m in synergy_matrices.values()])
    vmax = np.abs(all_values).max()
    vmin = -vmax

    last_im = None
    for digit in range(10):
        ax = axes[digit]
        matrix = synergy_matrices[digit]

        last_im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Digit {digit}", fontsize=14)
        ax.set_xlabel("Component j", fontsize=10)
        ax.set_ylabel("Component i", fontsize=10)

        # Set tick labels to component indices
        if n_alive <= 30:
            ax.set_xticks(range(n_alive))
            ax.set_xticklabels([str(c) for c in alive_components], fontsize=6, rotation=45)
            ax.set_yticks(range(n_alive))
            ax.set_yticklabels([str(c) for c in alive_components], fontsize=6)
        else:
            # Too many components, use subset of ticks
            tick_step = max(1, n_alive // 10)
            tick_positions = list(range(0, n_alive, tick_step))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [str(alive_components[i]) for i in tick_positions], fontsize=6, rotation=45
            )
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(alive_components[i]) for i in tick_positions], fontsize=6)

    # Add colorbar
    assert last_im is not None
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, label="Synergy I_{i,j}")
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle(
        f"Pairwise Component Synergy by Digit Class ({layer})\n"
        f"I_{{i,j}} = Δ_{{i,j}} - Δ_i - Δ_j (positive = superadditive, negative = subadditive)",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / f"synergy_heatmaps_{layer}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved synergy heatmaps to {output_path}")

    # Also save individual high-res plots
    for digit in range(10):
        fig, ax = plt.subplots(figsize=(12, 10))
        matrix = synergy_matrices[digit]

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Digit {digit} - Pairwise Component Synergy", fontsize=16)
        ax.set_xlabel("Component j", fontsize=12)
        ax.set_ylabel("Component i", fontsize=12)

        if n_alive <= 30:
            ax.set_xticks(range(n_alive))
            ax.set_xticklabels([str(c) for c in alive_components], fontsize=8, rotation=45)
            ax.set_yticks(range(n_alive))
            ax.set_yticklabels([str(c) for c in alive_components], fontsize=8)

        plt.colorbar(im, ax=ax, label="Synergy I_{i,j}")
        plt.tight_layout()

        digit_output_path = output_dir / f"synergy_digit_{digit}_{layer}.png"
        plt.savefig(digit_output_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved individual digit plots to {output_dir}")


def main(
    experiment_dir: str = "output/mnist_experiment_v2",
    checkpoint_step: int | None = None,
    layer: str = "fc1",
    n_samples_for_alive: int = 1000,
    output_dir: str | None = None,
):
    """Run synergy analysis on MNIST components.

    Args:
        experiment_dir: Path to experiment output directory
        checkpoint_step: Which checkpoint to load (None = latest)
        layer: Which layer to analyze ("fc1" or "fc2")
        n_samples_for_alive: Number of samples to use for determining alive components
        output_dir: Output directory for plots (default: experiment_dir/synergy_analysis)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set output directory
    if output_dir is None:
        output_path = Path(experiment_dir) / "synergy_analysis"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading component model...")
    component_model, target_model, config = load_component_model(
        experiment_dir, checkpoint_step, device
    )

    # Load MNIST test set
    print("Loading MNIST test set...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert to tensors
    test_images = []
    test_labels = []
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_images.append(image.view(-1))  # Flatten to 784
        test_labels.append(label)
    test_images = torch.stack(test_images)
    test_labels = np.array(test_labels)

    # Get alive components
    print(f"Finding alive components in {layer}...")
    alive_components = get_alive_components(
        component_model, test_images, layer=layer, n_samples=n_samples_for_alive
    )
    n_total = component_model.components[layer].V.shape[1]
    print(f"Found {len(alive_components)}/{n_total} alive components in {layer}")
    print(f"Alive component indices: {alive_components}")

    # Compute synergy matrices for each digit
    print("\nComputing synergy matrices...")
    synergy_matrices = {}
    for digit in range(10):
        print(f"\nProcessing digit {digit}...")
        synergy_matrices[digit] = compute_synergy_matrix(
            component_model,
            test_images,
            test_labels,
            alive_components,
            digit_class=digit,
            layer=layer,
        )

    # Save raw data
    save_data = {f"synergy_digit_{k}": v for k, v in synergy_matrices.items()}
    save_data["alive_components"] = np.array(alive_components)
    np.savez(output_path / f"synergy_data_{layer}.npz", **save_data)
    print(f"\nSaved raw synergy data to {output_path}/synergy_data_{layer}.npz")

    # Plot heatmaps
    print("\nGenerating plots...")
    plot_synergy_heatmaps(synergy_matrices, alive_components, output_path, layer)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for digit in range(10):
        matrix = synergy_matrices[digit]
        # Get off-diagonal elements only
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        print(
            f"Digit {digit}: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, "
            f"max={off_diag.max():.4f}, min={off_diag.min():.4f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
