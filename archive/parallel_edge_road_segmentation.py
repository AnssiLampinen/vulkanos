"""
parallel_edge_road_segmentation.py
----------------------------------
2-Layer CNN Pipeline for Satellite Road Extraction:

Layer 1: Deep Edge Detection CNN (DexiNed)
  - Extracts continuous multi-scale edge and boundary probability maps.

Layer 2: Curved Parallel Edge Corridor CNN
  - Scans across multiple road widths and fine-grained orientation angles (16 directions)
    to accurately trace both straight and curving roads.
  - Directional paired-edge geometry: detects complementary parallel boundaries.
  - Transverse Wall Suppression: penalizes perpendicular edges crossing the corridor (eliminating building walls/rooms).
  - Isotropic Clutter Normalization: suppresses dense multi-directional rooftop/urban clutter.
  - Morphological Path Continuity: removes small isolated non-road blobs while retaining continuous road networks.
"""

from huggingface_hub import hf_hub_download
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Import DexiNed CNN architecture from edge_detection_test.py
from edge_detection_test import DexiNed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Layer 2: Curved Parallel Edge Corridor CNN
# =====================================================================

class CurvedParallelRoadCNN(nn.Module):
    """
    CNN Layer that evaluates directional edge pairing, parallelism, curvature,
    and suppresses building polygons.
    """
    def __init__(self, num_angles: int = 16, road_widths: list = [10, 16, 24, 32, 42]):
        super().__init__()
        self.num_angles = num_angles
        self.road_widths = road_widths

    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_map: Tensor [B, 1, H, W] of continuous edge probabilities from Layer 1.
        Returns:
            Tensor [B, 1, H, W] containing classified road corridor likelihoods.
        """
        # Suppress weak background edge noise
        cleaned_edges = F.relu(edge_map - 0.18)

        # 1. Compute local isotropic edge clutter (dense urban building indicator)
        clutter_ksize = 31
        clutter_kernel = torch.ones(1, 1, clutter_ksize, clutter_ksize, device=edge_map.device) / (clutter_ksize ** 2)
        clutter_density = F.conv2d(cleaned_edges, clutter_kernel, padding=clutter_ksize // 2)

        total_road_accumulator = torch.zeros_like(edge_map)

        for width in self.road_widths:
            half_w = width / 2.0
            # Kernel size tailored to width with adequate support
            ksize = int(max(width * 2.2, 25))
            if ksize % 2 == 0:
                ksize += 1
            center = ksize // 2
            pad = center

            y, x = torch.meshgrid(
                torch.arange(ksize, dtype=torch.float32, device=edge_map.device) - center,
                torch.arange(ksize, dtype=torch.float32, device=edge_map.device) - center,
                indexing="ij"
            )

            width_accumulator = torch.zeros_like(edge_map)

            for a in range(self.num_angles):
                theta = a * math.pi / self.num_angles
                u = x * math.cos(theta) + y * math.sin(theta)   # Tangent along road axis
                v = -x * math.sin(theta) + y * math.cos(theta)  # Normal across road axis

                # Moderate longitudinal span to accommodate road curvature without straight-line attenuation
                L = max(width * 0.9, 10.0)
                long_weight = torch.exp(-0.5 * (u / L) ** 2)

                sigma_edge = 1.4
                # Left & right parallel boundary probes
                k_left = torch.exp(-0.5 * ((v - half_w) / sigma_edge) ** 2) * long_weight
                k_right = torch.exp(-0.5 * ((v + half_w) / sigma_edge) ** 2) * long_weight

                # Interior corridor probe (checks if corridor is clear)
                k_inside = (torch.abs(v) <= half_w * 0.70).float() * long_weight

                # Transverse wall probe (checks for perpendicular edges cutting across corridor, typical of building rooms/boxes)
                sigma_trans = 1.4
                k_transverse = torch.exp(-0.5 * (u / sigma_trans) ** 2) * (torch.abs(v) <= half_w * 0.95).float()

                # Normalize probe kernels
                k_left = (k_left / (k_left.sum() + 1e-6)).view(1, 1, ksize, ksize)
                k_right = (k_right / (k_right.sum() + 1e-6)).view(1, 1, ksize, ksize)
                k_inside = (k_inside / (k_inside.sum() + 1e-6)).view(1, 1, ksize, ksize)
                k_transverse = (k_transverse / (k_transverse.sum() + 1e-6)).view(1, 1, ksize, ksize)

                resp_left = F.conv2d(cleaned_edges, k_left, padding=pad)
                resp_right = F.conv2d(cleaned_edges, k_right, padding=pad)
                resp_inside = F.conv2d(cleaned_edges, k_inside, padding=pad)
                resp_transverse = F.conv2d(cleaned_edges, k_transverse, padding=pad)

                # Both parallel boundaries must be present
                parallel_pair_score = torch.minimum(resp_left, resp_right)

                # Penalize transverse blocking walls (buildings) and internal clutter
                corridor = F.relu(parallel_pair_score - 0.70 * resp_transverse - 0.30 * resp_inside)

                # Fill the continuous road surface between the two boundary edges
                fill_k = torch.exp(-0.5 * (v / (half_w * 0.65)) ** 2) * long_weight
                fill_k = (fill_k / (fill_k.sum() + 1e-6)).view(1, 1, ksize, ksize)
                road_fill = F.conv2d(corridor, fill_k, padding=pad)

                width_accumulator = torch.maximum(width_accumulator, road_fill)

            total_road_accumulator = torch.maximum(total_road_accumulator, width_accumulator)

        # 2. Apply Isotropic Clutter Suppression:
        # Buildings have edges in all directions. Roads have clean anisotropic corridors.
        clutter_suppression_factor = 1.0 / (1.0 + 8.0 * (clutter_density ** 1.3))
        filtered_road = total_road_accumulator * clutter_suppression_factor

        # Normalize score into [0, 1] range
        min_val = filtered_road.min()
        max_val = filtered_road.max()
        normalized_road = (filtered_road - min_val) / (max_val - min_val + 1e-6)

        return normalized_road


def filter_connected_components(binary_mask: np.ndarray, min_size: int = 120) -> np.ndarray:
    """
    Morphological network filter: eliminates small isolated building fragments
    while preserving continuous road segments.
    """
    labeled_array, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask
    
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_array]
    
    cleaned_mask = binary_mask.copy()
    cleaned_mask[too_small_mask] = 0
    return cleaned_mask


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    print("=" * 65)
    print("2-LAYER CNN: EDGE DETECTION + CURVED PARALLEL ROAD EXTRACTION")
    print("=" * 65)

    # 1. Load Layer 1: DexiNed Edge Detection CNN
    print("[Layer 1] Loading DexiNed Edge CNN...")
    ckpt_path = hf_hub_download(repo_id="kornia/dexined", filename="DexiNed_BIPED_10.pth")
    edge_model = DexiNed()
    edge_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    edge_model.to(device).eval()
    print("Layer 1 ready.")

    # 2. Instantiate Layer 2: Curved Parallel Edge Corridor CNN
    print("[Layer 2] Initializing Curved Parallel Edge Corridor CNN...")
    corridor_model = CurvedParallelRoadCNN(
        num_angles=16,
        road_widths=[10, 16, 24, 32, 42]
    ).to(device).eval()
    print("Layer 2 ready.")

    # 3. Load & Preprocess Satellite Image
    image_path = "satellite_image.tif"
    image = Image.open(image_path).convert("RGB")
    crop_size = 512
    image_cropped = image.crop((0, 0, crop_size, crop_size))

    tensor_img = TF.to_tensor(image_cropped).unsqueeze(0).to(device)
    mean = torch.tensor([103.53, 116.28, 123.675], device=device).view(1, 3, 1, 1)
    inp = (tensor_img * 255.0) - mean

    # 4. Forward Pass through Layer 1 & Layer 2
    print("\nExecuting Layer 1 (DexiNed Edge Extraction)...")
    with torch.no_grad():
        fused_out, _ = edge_model(inp)
        edge_prob = torch.sigmoid(fused_out)  # [1, 1, H, W]

    print("Executing Layer 2 (Extracting Curved Parallel Road Corridors)...")
    with torch.no_grad():
        road_corridor_score = corridor_model(edge_prob)

    edge_map_np = edge_prob.squeeze().cpu().numpy()
    road_score_np = road_corridor_score.squeeze().cpu().numpy()

    # Threshold road corridor responses and remove small isolated non-road fragments
    threshold = 0.28
    initial_mask = (road_score_np > threshold).astype(np.uint8)
    road_mask = filter_connected_components(initial_mask, min_size=120)

    # 5. Visual Display
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_cropped)
    axes[0].set_title("Input Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(edge_map_np, cmap="gray")
    axes[1].set_title("Layer 1: DexiNed CNN Edges")
    axes[1].axis("off")

    axes[2].imshow(road_score_np, cmap="inferno")
    axes[2].set_title("Layer 2: Curved Corridor Response")
    axes[2].axis("off")

    # Overlay classified road corridors in bright orange/red
    axes[3].imshow(image_cropped)
    axes[3].imshow(np.ma.masked_where(road_mask == 0, road_mask), cmap="autumn", alpha=0.6)
    axes[3].set_title(f"Classified Roads (Curved & Anti-Building)")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = "curved_road_segmentation_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved segmentation visualization to: {output_path}")
    print("Displaying 2-layer pipeline results...")
    plt.show()


if __name__ == "__main__":
    main()


