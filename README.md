# DSPC-Net: Dual-Stream Phase-Consistent Network for Flood Detection

This repository provides the minimal core implementation, inference code, and specific loss function definitions for **DSPC-Net**, designed to satisfy reproducibility requirements for our research paper. It allows reviewers and researchers to quickly verify the model's performance on representative SAR image samples.

## Overview

DSPC-Net is an advanced deep learning architecture tailored for accurate flood detection in SAR imagery, effectively handling complex scattering characteristics and blurred boundaries.

**Key Innovations:**
* **Dual-Stream Encoder**: Captures multi-scale spatial details and structural features simultaneously.
* **LCSC Module (Local Context Skip Connection)**: Optimizes feature fusion by strictly maintaining high-resolution base sizes during skip connections, avoiding redundant upsampling distortions.
* **Optimized Loss Function**: Employs a meticulously tuned joint loss strategy (`Total Loss = 0.8 * CFCE + 0.2 * Dice`) to combat severe foreground-background class imbalance.

## Environment Requirements

| Library             | Version (Recommended)      |
| ------------------- | -------------------------- |
| Python              | 3.9 / 3.10                 |
| torch               | ≥ 1.10 (CUDA optional)     |
| numpy               | latest                     |
| pillow              | latest                     |
| tifffile            | latest                     |

## Repository Structure

```text
├── main.py                    # Definition of the core DSPC-Net architecture
├── custom/
│   ├── __init__.py            # Python package initialization
│   └── unet_dualstream.py     # Underlying building blocks (Encoder, LCSC Fuser)
├── loss.py                    # Implementation of the 0.8 CFCE + 0.2 Dice joint loss
├── run_demo.py                # Standard single-image inference and visualization script
└── demo_data/                 # Representative SAR images and ground truth masks
    ├── images/
    │   └── sample_01.tif      # Input SAR image sample
    └── labels/
        └── sample_01.tif      # Corresponding ground truth mask sample
```

## Getting Started

### 1. Environment Setup
You can install the required dependencies directly using `pip`:

```bash
pip install torch numpy Pillow tifffile
```

### 2. Quick Inference Demo
To process the provided demo samples and generate a qualitative comparison grid (SAR vs. Ground Truth vs. DSPC-Net), simply run the evaluation script. 

By default, the script will process the sample data in the `demo_data/` folder:
```bash
python run_demo.py
```

*Customizing inputs:* You can also test the network on specific SAR images by passing custom arguments:
```bash
python run_demo.py \
    --image_path ./demo_data/images/your_image.tif \
    --label_path ./demo_data/labels/your_label.tif
```

### Visual Output Guide
The script will output a high-resolution comparison image to the `./outputs` directory. We employ an academic color mapping engine to visualize pixel-level accuracy directly:
* **Blue (True Positive):** Correctly detected water bodies.
* **Red (False Positive):** Over-predicted areas (Noise/Shadows).
* **Yellow (False Negative):** Missed topological details (Disconnected water bodies).
