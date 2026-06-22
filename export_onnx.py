"""
export_onnx.py
--------------
Exports a trained YOLOv8 PPE detection model to ONNX format
for optimized server-side inference.

Usage:
    python export_onnx.py --weights best.pt --imgsz 640 --batch 1

Requirements:
    pip install ultralytics onnx onnxruntime
"""

import argparse
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PPE model to ONNX")
    parser.add_argument("--weights",  type=str, default="best.pt",
                        help="Path to trained YOLOv8 .pt weights file")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Input image size (square, default 640)")
    parser.add_argument("--batch",   type=int, default=1,
                        help="Batch size for export (default 1 for server inference)")
    parser.add_argument("--opset",   type=int, default=17,
                        help="ONNX opset version (default 17)")
    parser.add_argument("--simplify",action="store_true", default=True,
                        help="Run onnx-simplifier after export")
    parser.add_argument("--output",  type=str, default="ppe_detector.onnx",
                        help="Output ONNX file name")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(args) -> Path:
    print(f"\n{'='*55}")
    print("  PPE Detector — YOLOv8 → ONNX Export")
    print(f"{'='*55}\n")

    print(f"[1/4]  Loading weights : {args.weights}")
    model = YOLO(args.weights)

    print(f"[2/4]  Exporting to ONNX (opset {args.opset}, imgsz {args.imgsz}) ...")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,          # static shapes → faster server inference
    )
    export_path = Path(export_path)

    # Rename to user-specified output name
    final_path = export_path.parent / args.output
    export_path.rename(final_path)
    print(f"         Saved → {final_path}")

    return final_path


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_onnx(onnx_path: Path):
    print(f"\n[3/4]  Validating ONNX model ...")

    # 1. Schema check
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("         ✓ ONNX schema valid")

    # 2. Runtime smoke-test
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options)

    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape   # [batch, C, H, W]
    # Replace dynamic dims (strings) with 1 / 640
    resolved = [d if isinstance(d, int) else 1 for d in input_shape]

    dummy = torch.zeros(resolved).numpy()
    t0 = time.perf_counter()
    _ = session.run(None, {input_name: dummy})
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"         ✓ Inference smoke-test passed  ({latency_ms:.1f} ms)")

    # 3. File size
    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"         ✓ Model size : {size_mb:.2f} MB")

    return latency_ms, size_mb


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(onnx_path: Path, latency_ms: float, size_mb: float):
    print(f"\n[4/4]  Export Summary")
    print(f"{'─'*45}")
    print(f"  Output file   : {onnx_path}")
    print(f"  Model size    : {size_mb:.2f} MB")
    print(f"  Latency (CPU) : {latency_ms:.1f} ms  (single image, no warm-up)")
    print(f"\n  Next steps:")
    print(f"    • Run quantize_onnx.py to reduce size further")
    print(f"    • Deploy with serve_onnx.py (FastAPI + ORT)")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    onnx_path  = export_to_onnx(args)
    latency, size = validate_onnx(onnx_path)
    print_summary(onnx_path, latency, size)
