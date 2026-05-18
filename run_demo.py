import os
import tifffile
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Import the core network architecture
from main import DSPCNet  # Ensure your network definition file is named dspc_net.py and class is DSPCNet


# ---------------------------------------------------------
# Academic Color Mapping Engine (Pixel-level TP/FP/FN)
# ---------------------------------------------------------
def generate_gt_rgb(gt_mask):
    h, w = gt_mask.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    COLOR_BLUE = np.array([0, 51, 153], dtype=np.uint8)
    rgb[gt_mask == 1] = COLOR_BLUE
    return Image.fromarray(rgb)


def generate_eval_rgb(pred_mask, gt_mask):
    h, w = pred_mask.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)

    COLOR_TP = np.array([0, 51, 153], dtype=np.uint8)  # True Positive: Blue
    COLOR_FP = np.array([178, 0, 0], dtype=np.uint8)  # False Positive: Red
    COLOR_FN = np.array([204, 153, 0], dtype=np.uint8)  # False Negative: Yellow

    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)

    rgb[tp_mask] = COLOR_TP
    rgb[fp_mask] = COLOR_FP
    rgb[fn_mask] = COLOR_FN

    return Image.fromarray(rgb)


# ---------------------------------------------------------
# Grid Layout Engine
# ---------------------------------------------------------
def create_comparison_grid(sar_img, gt_img, pred_img, panel_titles, output_path, gap=15, border_width=1):
    STD_W, STD_H = sar_img.width, sar_img.height
    all_images = [sar_img, gt_img, pred_img]
    num_panels = len(all_images)
    LABEL_H = int(STD_H * 0.15)

    grid_w = (STD_W * num_panels) + (gap * (num_panels - 1))
    grid_h = STD_H + LABEL_H

    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)

    # Paste images and draw borders
    for i, img in enumerate(all_images):
        paste_x = i * (STD_W + gap)
        grid_img.paste(img, (paste_x, 0))
        if border_width > 0:
            draw.rectangle(
                [paste_x, 0, paste_x + STD_W - 1, STD_H - 1],
                outline="black", width=border_width
            )

    # Configure font and titles
    font_size = int(LABEL_H * 0.55)
    main_font = ImageFont.load_default()
    txt_color = "black"

    for i, title in enumerate(panel_titles):
        letter_only = title.split(')')[0] + ')'
        text_only = title.split(')')[1].strip()

        try:
            lettering_w = draw.textlength(letter_only, font=main_font)
            gap_w = draw.textlength(" ", font=main_font)
            text_w = draw.textlength(text_only, font=main_font)
        except AttributeError:
            lettering_w = main_font.getsize(letter_only)[0]
            gap_w = main_font.getsize(" ")[0]
            text_w = main_font.getsize(text_only)[0]

        total_w = lettering_w + gap_w + text_w
        panel_center_x = (i * (STD_W + gap)) + (STD_W // 2)
        base_y = grid_h - LABEL_H

        start_draw_x = panel_center_x - (total_w // 2)
        draw.text((start_draw_x, base_y + int(LABEL_H * 0.3)), letter_only, fill=txt_color, font=main_font)
        draw.text((start_draw_x + lettering_w + gap_w, base_y + int(LABEL_H * 0.3)), text_only, fill=txt_color,
                  font=main_font)

    grid_img.save(output_path, dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser(description="DSPC-Net Single Image Inference Script")
    parser.add_argument('--weight_path', type=str, default='./weights/dspc_net_best.pt',
                        help='Path to the model weights')
    parser.add_argument('--image_path', type=str, default='./demo_data/images/sample_01.tif',
                        help='Path to the input SAR image (.tif)')
    parser.add_argument('--label_path', type=str, default='./demo_data/labels/sample_01.tif',
                        help='Path to the ground truth mask (.tif)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save visualization results')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[*] Initializing DSPC-Net for Inference...")
    model = DSPCNet(in_channels=2, seg_out_channels=2).to(device)

    if not os.path.exists(args.weight_path):
        print(f"[!] Weight file not found at {args.weight_path}.")
        return

    # Load weights
    ckpt = torch.load(args.weight_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[√] Weights loaded successfully.")

    if not os.path.exists(args.image_path) or not os.path.exists(args.label_path):
        print(
            f"[!] Input image or label not found. Please check the paths:\n  Image: {args.image_path}\n  Label: {args.label_path}")
        return

    print(f"\n[*] Processing sample: {Path(args.image_path).name} ...")
    panel_titles = ["(a) SAR", "(b) Ground Truth", "(c) DSPC-Net (Ours)"]

    # Load single image and label
    x_np = tifffile.imread(args.image_path)
    gt_np = tifffile.imread(args.label_path)

    x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    # Extract grayscale image for visualization
    sar_mean = x_np.mean(axis=0)
    p_min, p_max = np.percentile(sar_mean, (1, 99))
    sar_clipped = np.clip(sar_mean, p_min, p_max) if p_max > p_min else sar_mean
    x_grayscale = ((sar_clipped - p_min) / (p_max - p_min + 1e-5) * 255).astype(np.uint8)
    sar_img_pil = Image.fromarray(x_grayscale).convert('RGB')

    gt_img_pil = generate_gt_rgb(gt_np)

    # Inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            out = model(x_tensor)
            logits = out["seg"] if isinstance(out, dict) else out
            pred_class = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    eval_img_pil = generate_eval_rgb(pred_mask=pred_class, gt_mask=gt_np)

    # Save results
    stem = Path(args.image_path).stem
    out_file = Path(args.output_dir) / f"{stem}_DSPC_comparison.png"
    create_comparison_grid(sar_img_pil, gt_img_pil, eval_img_pil, panel_titles, out_file)

    print(f"\n[√] Inference complete. Result saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()