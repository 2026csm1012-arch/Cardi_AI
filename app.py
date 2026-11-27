"""
Gradio UI for Echocardiography Ejection Fraction (EF) Prediction using Mamba (GapSSM).

Run locally:
    python app.py

Run on Kaggle (creates a public shareable link):
    !pip install -q gradio
    !python app.py --share --checkpoint /kaggle/working/cardiac_runs/mamba/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure src/ is importable without requiring PYTHONPATH=src
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac.data.dataset import load_video, normalize
from cardiac.data.missing import handle_missing, make_mask
from cardiac.data.sampling import clip_indices
from cardiac.evaluation.runner import mc_predict
from cardiac.models.factory import build_model


_MODEL_CACHE = {}


def load_mamba_checkpoint(ckpt_path: str):
    """Load and cache trained GapSSM (Mamba) checkpoint."""
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {path}. "
            "Train the Mamba model first or provide a valid best.pt path."
        )

    cache_key = str(path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]

    model = build_model({**cfg, "pretrained": False}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _MODEL_CACHE[cache_key] = (model, cfg, device)
    return model, cfg, device


def classify_ef(ef: float) -> str:
    """Standard clinical LVEF category."""
    if ef >= 55.0:
        return "Normal (EF >= 55%)"
    if ef >= 45.0:
        return "Mildly Reduced (45% - 54%)"
    if ef >= 30.0:
        return "Moderately Reduced (30% - 44%)"
    return "Severely Reduced (EF < 30%)"


def make_timeline_figure(frames: np.ndarray, mask: np.ndarray):
    """Render a strip of sampled frames showing observed vs missing frames."""
    n_show = min(8, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(12, 2.6))
    if n_show == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        img = frames[idx, 0] if frames.ndim == 4 else frames[idx]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        present = bool(mask[idx]) if idx < len(mask) else True
        status = "OK" if present else "DROPPED"
        color = "#16a34a" if present else "#dc2626"
        ax.set_title(f"t={idx} ({status})", fontsize=8, color=color, fontweight="bold", pad=4)
        ax.axis("off")

    fig.suptitle("Sampled Frame Timeline (Kept vs Dropped)", fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return fig


def find_default_checkpoint() -> str:
    """Find best.pt in checkpoints/, runs/, or /kaggle/working/cardiac_runs/."""
    candidates = [
        Path("checkpoints/best.pt"),
        Path("checkpoints/mamba/best.pt"),
        Path("/kaggle/working/cardiac_runs/mamba/best.pt"),
        Path("runs/mamba/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        pts = sorted(ckpt_dir.rglob("*.pt"))
        if pts:
            return str(pts[0])
    return "checkpoints/best.pt"


def predict_video(
    video_file: str,
    checkpoint_path: str,
    checkpoint_upload,
    missing_ratio: float,
    missing_pattern: str,
    handling: str,
    mc_passes: int,
):
    if not video_file:
        raise gr.Error("Please upload an echocardiography video (.avi or .mp4).")

    active_ckpt = checkpoint_upload if checkpoint_upload else checkpoint_path
    model, cfg, device = load_mamba_checkpoint(active_ckpt)
    length = int(cfg.get("length", 32))
    stride = int(cfg.get("stride", 2))
    size = int(cfg.get("size", 112))

    # Load and sample clip identically to training/evaluation
    raw_frames = load_video(video_file, size=size)
    indices = clip_indices(len(raw_frames), length, stride, rng=None)
    sampled = raw_frames[indices]
    times = (indices - indices[0]).astype(np.float32) / stride

    if missing_ratio > 0.0:
        mask = make_mask(len(indices), float(missing_ratio), missing_pattern, seed=42)
    else:
        mask = np.ones(len(indices), dtype=bool)

    proc_frames, proc_mask, proc_times = handle_missing(sampled, mask, times, handling)
    video_tensor = torch.from_numpy(normalize(proc_frames)).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(proc_mask).unsqueeze(0).to(device)
    times_tensor = torch.from_numpy(proc_times).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    if mc_passes >= 2:
        mean_ef, std_ef = mc_predict(model, video_tensor, mask_tensor, times_tensor, int(mc_passes))
        ef_val = float(mean_ef.item())
        unc_val = float(std_ef.item())
    else:
        with torch.no_grad():
            ef_val = float((100.0 * model(video_tensor, mask_tensor, times_tensor)).item())
        unc_val = 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fig = make_timeline_figure(proc_frames, proc_mask)
    category = classify_ef(ef_val)

    summary = (
        f"### Predicted LVEF: **{ef_val:.2f}%** (± {unc_val:.2f}%)\n"
        f"- **Clinical Category:** {category}\n"
        f"- **Frames Kept:** {int(proc_mask.sum())} / {len(proc_mask)} "
        f"(Missing Ratio: {missing_ratio:.0%}, Pattern: `{missing_pattern}`, Handling: `{handling}`)\n"
        f"- **Inference Time:** {elapsed_ms:.1f} ms on `{device.type.upper()}` (`mc_passes={int(mc_passes)}`)"
    )
    return summary, fig


def build_ui(default_ckpt: str) -> gr.Blocks:
    with gr.Blocks(title="Cardi_AI — Mamba (GapSSM) EF Predictor") as demo:
        gr.Markdown(
            "# Cardi_AI: Echocardiography EF Prediction (Mamba / GapSSM)\n"
            "Upload an apical 4-chamber echocardiography video (`.avi` / `.mp4`) to predict "
            "**Left-Ventricular Ejection Fraction (EF)** and optionally test missing-frame robustness."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Upload Echocardiography Video (.avi / .mp4)",
                    file_types=[".avi", ".mp4", ".mov", ".mkv"],
                    type="filepath",
                )
                ckpt_input = gr.Textbox(
                    label="Mamba Checkpoint Path (drop best.pt into checkpoints/)",
                    value=default_ckpt,
                )
                ckpt_upload = gr.File(
                    label="Or Upload Checkpoint (.pt)",
                    file_types=[".pt", ".pth"],
                    type="filepath",
                )
                with gr.Accordion("Missing-Frame Simulation & Uncertainty", open=False):
                    ratio_slider = gr.Slider(
                        0.0, 0.75, value=0.0, step=0.05, label="Missing Frame Ratio"
                    )
                    pattern_dd = gr.Dropdown(
                        ["random", "block"], value="random", label="Missing Pattern"
                    )
                    handling_dd = gr.Dropdown(
                        ["zero", "removal", "repeat", "interpolate"],
                        value="zero",
                        label="Input Handling",
                    )
                    mc_slider = gr.Slider(
                        1, 20, value=10, step=1, label="MC Dropout Passes (1 = single pass)"
                    )
                run_btn = gr.Button("Predict Ejection Fraction", variant="primary")

            with gr.Column(scale=1):
                result_md = gr.Markdown(label="Prediction Summary")
                timeline_plot = gr.Plot(show_label=False)

        run_btn.click(
            fn=predict_video,
            inputs=[video_input, ckpt_input, ckpt_upload, ratio_slider, pattern_dd, handling_dd, mc_slider],
            outputs=[result_md, timeline_plot],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Cardi_AI Mamba Gradio UI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=find_default_checkpoint(),
        help="Path to trained Mamba best.pt checkpoint",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link (useful on Kaggle)")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    ui = build_ui(args.checkpoint)
    ui.launch(server_port=args.port, share=args.share)
