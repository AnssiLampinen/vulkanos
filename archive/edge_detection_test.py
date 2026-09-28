"""
edge_detection_test.py
----------------------
Deep Learning Edge Detection Test using DexiNed (Dense Extreme Inception Network).

DexiNed is a state-of-the-art fully convolutional network (CNN) trained on high-quality 
edge datasets (BIPED) that predicts crisp multi-scale edge maps and structural contours 
without requiring post-processing or manual gradient filters.
"""

from collections import OrderedDict
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# DexiNed CNN Architecture
# =====================================================================

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int) -> None:
        super().__init__(
            OrderedDict([
                ("relu1", nn.ReLU(inplace=True)),
                ("conv1", nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True)),
                ("norm1", nn.BatchNorm2d(out_features)),
                ("relu2", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True)),
                ("norm2", nn.BatchNorm2d(out_features)),
            ])
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x1, x2 = x[0], x[1]
        x3 = x1
        for mod in self:
            x3 = mod(x3)
        return [0.5 * (x3 + x2), x2]


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module(f"denselayer{(i + 1)}", layer)
            input_features = out_features

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x_out = x
        for mod in self:
            x_out = mod(x_out)
        return x_out


class UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        self.constant_features = 16
        layers = nn.ModuleList([])
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2**up_scale
            pad = all_pads[up_scale]
            out_features = 1 if i == up_scale - 1 else self.constant_features
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, out_shape: list[int]) -> torch.Tensor:
        out = self.features(x)
        return F.interpolate(out, out_shape, mode="bilinear", align_corners=False)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_bs: bool = True) -> None:
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Sequential):
    def __init__(self, in_features: int, mid_features: int, out_features=None, stride: int = 1, use_act: bool = True) -> None:
        super().__init__()
        if out_features is None:
            out_features = mid_features
        self.add_module("conv1", nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride))
        self.add_module("bn1", nn.BatchNorm2d(mid_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(mid_features, out_features, 3, padding=1))
        self.add_module("bn2", nn.BatchNorm2d(out_features))
        if use_act:
            self.add_module("relu2", nn.ReLU(inplace=True))


class DexiNed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Skip connections
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        # Multi-scale upsampling blocks
        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False)

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        block_5_pre_dense = self.pre_dense_5(block_4_down)
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        out_shape = x.shape[-2:]
        out_1 = self.up_block_1(block_1, out_shape)
        out_2 = self.up_block_2(block_2, out_shape)
        out_3 = self.up_block_3(block_3, out_shape)
        out_4 = self.up_block_4(block_4, out_shape)
        out_5 = self.up_block_5(block_5, out_shape)
        out_6 = self.up_block_6(block_6, out_shape)
        return [out_1, out_2, out_3, out_4, out_5, out_6]

    def forward(self, x: torch.Tensor):
        features = self.get_features(x)
        block_cat = torch.cat(features, dim=1)
        fused = self.block_cat(block_cat)
        return fused, features


# =====================================================================
# Main Edge Detection Pipeline
# =====================================================================

def main():
    print("Loading DexiNed edge detection model...")
    checkpoint_path = hf_hub_download(
        repo_id="kornia/dexined", 
        filename="DexiNed_BIPED_10.pth"
    )
    
    model = DexiNed()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("Model loaded successfully.")

    # 1. Load satellite image crop
    image_path = "satellite_image.tif"
    image = Image.open(image_path).convert("RGB")
    crop_size = 512
    image_cropped = image.crop((0, 0, crop_size, crop_size))

    # 2. Preprocess (BIPED mean subtraction in [0, 255] color space)
    tensor_img = TF.to_tensor(image_cropped).unsqueeze(0).to(device)
    mean = torch.tensor([103.53, 116.28, 123.675], device=device).view(1, 3, 1, 1)
    inp = (tensor_img * 255.0) - mean

    # 3. CNN Edge Inference
    print("Running CNN edge detection inference...")
    with torch.no_grad():
        fused_out, side_features = model(inp)
        
        # Fused edge map
        fused_prob = torch.sigmoid(fused_out).squeeze().cpu().numpy()
        
        # Average across all 6 intermediate multi-scale feature hierarchies
        side_probs = [torch.sigmoid(f).squeeze().cpu().numpy() for f in side_features]
        multiscale_avg = np.mean(side_probs, axis=0)

    # 4. Binary Edge Map (Thresholded)
    threshold = 0.40
    binary_edges = (fused_prob > threshold).astype(np.uint8)

    # 5. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_cropped)
    axes[0].set_title("Input Satellite Crop")
    axes[0].axis("off")

    axes[1].imshow(fused_prob, cmap="gray")
    axes[1].set_title("DexiNed Fused Edges (CNN)")
    axes[1].axis("off")

    axes[2].imshow(multiscale_avg, cmap="magma")
    axes[2].set_title("Multi-Scale Edge Response")
    axes[2].axis("off")

    # Overlay edges in cyan over the satellite image
    axes[3].imshow(image_cropped)
    axes[3].imshow(np.ma.masked_where(binary_edges == 0, binary_edges), cmap="cool", alpha=0.7)
    axes[3].set_title(f"Detected Edges Overlay (thr={threshold})")
    axes[3].axis("off")

    plt.tight_layout()
    print("Displaying edge detection results...")
    plt.show()


if __name__ == "__main__":
    main()

