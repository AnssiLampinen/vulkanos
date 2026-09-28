from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------

# 1. Massachusetts Roads Dataset Benchmark U-Net Architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return torch.sigmoid(self.conv_final(d1))


# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
print("Loading road segmentation models...")

# Model A: Massachusetts Roads Dataset Pre-trained U-Net
mass_model_path = hf_hub_download(
    repo_id="teohyc/Satellite-Road-Segmentation-UNet", 
    filename="best_road_seg_unet.pth"
)
unet_mass_model = UNet(in_channels=3, out_channels=1)
unet_mass_model.load_state_dict(torch.load(mass_model_path, map_location=device))
unet_mass_model.to(device).eval()

# Model B: UnetPlusPlus with EfficientNet backbone
upp_model_path = hf_hub_download(
    repo_id="manish2607/road_extrcation_model_BAH", 
    filename="road_model_Deployment.pt"
)
upp_model = torch.jit.load(upp_model_path, map_location=device)
upp_model.to(device).eval()

# ---------------------------------------------------------
# Load and Preprocess Image
# ---------------------------------------------------------
image = Image.open("satellite_image.tif").convert("RGB")
crop_size = 512
image_cropped = image.crop((0, 0, crop_size, crop_size))
tensor_img = F.to_tensor(image_cropped).unsqueeze(0).to(device)


# ---------------------------------------------------------
# Multi-Scale & Test-Time Augmentation (TTA) Inference
# ---------------------------------------------------------
def predict_with_tta(model, x, is_prob=False):
    """Predicts with horizontal and vertical flip augmentations to reduce noise."""
    preds = []
    # 1. Normal
    with torch.no_grad():
        out = model(x)
        preds.append(out if is_prob else torch.sigmoid(out))
    # 2. Horizontal flip
    x_h = torch.flip(x, dims=[3])
    with torch.no_grad():
        out_h = model(x_h)
        p_h = out_h if is_unet_pred(out_h, is_prob) else torch.sigmoid(out_h)
        preds.append(torch.flip(p_h, dims=[3]))
    # 3. Vertical flip
    x_v = torch.flip(x, dims=[2])
    with torch.no_grad():
        out_v = model(x_v)
        p_v = out_v if is_unet_pred(out_v, is_prob) else torch.sigmoid(out_v)
        preds.append(torch.flip(p_v, dims=[2]))

    return torch.stack(preds).mean(dim=0)


def is_unet_pred(out, is_prob):
    return is_prob


print("Running multi-scale ensemble inference...")

# 1. Base scale predictions
prob_upp_1x = predict_with_tta(upp_model, tensor_img, is_prob=False)
prob_mass_1x = predict_with_tta(unet_mass_model, tensor_img, is_prob=True)

# 2. Multi-scale zoom (catches thin street corridors, alleys, and connectors)
tensor_upscaled = F_nn.interpolate(tensor_img, scale_factor=1.5, mode="bilinear", align_corners=False)
prob_upp_upscaled = predict_with_tta(upp_model, tensor_upscaled, is_prob=False)
prob_upp_upscaled = F_nn.interpolate(prob_upp_upscaled, size=(crop_size, crop_size), mode="bilinear", align_corners=False)

# ---------------------------------------------------------
# Ensemble & Post-Processing
# ---------------------------------------------------------
# Combine multi-scale UnetPlusPlus with Massachusetts U-Net for superior road connectivity
upp_combined = (prob_upp_1x + prob_upp_upscaled) / 2.0
ensemble_prob = 0.70 * upp_combined + 0.30 * prob_mass_1x

prob_map = ensemble_prob.squeeze().cpu().numpy()
prediction_mask = (prob_map > 0.30).astype(np.uint8)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(image_cropped)
axes[0].set_title("Input Satellite Image")
axes[0].axis("off")

axes[1].imshow(prob_map, cmap="inferno")
axes[1].set_title("Road Probability Heatmap")
axes[1].axis("off")

axes[2].imshow(prediction_mask, cmap="gray")
axes[2].set_title("Predicted Road Mask (thr=0.30)")
axes[2].axis("off")

# Red/yellow road overlay on original satellite image
axes[3].imshow(image_cropped)
axes[3].imshow(np.ma.masked_where(prediction_mask == 0, prediction_mask), cmap="autumn", alpha=0.6)
axes[3].set_title("Road Network Overlay")
axes[3].axis("off")

plt.tight_layout()
plt.show()
plt.show()