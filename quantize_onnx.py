"""
quantize_onnx.py
----------------
Applies INT8 static quantization to the exported PPE detector ONNX model,
reducing model size by ~4x and improving CPU throughput for server deployment.

Usage:
    python quantize_onnx.py --model ppe_detector.onnx --calib-images ./calib_images

Requirements:
    pip install onnxruntime onnx opencv-python numpy
"""

import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="INT8 static quantization for YOLOv8 ONNX PPE detector"
    )
    parser.add_argument("--model",        type=str, default="ppe_detector.onnx",
                        help="Input float32 ONNX model")
    parser.add_argument("--output",       type=str, default="ppe_detector_int8.onnx",
                        help="Output INT8 quantized ONNX model")
    parser.add_argument("--calib-images", type=str, default="./calib_images",
                        help="Directory of calibration images (JPG/PNG)")
    parser.add_argument("--imgsz",        type=int, default=640,
                        help="Model input image size")
    parser.add_argument("--num-calib",    type=int, default=50,
                        help="Max calibration images to use")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class PPECalibrationReader(CalibrationDataReader):
    """Feeds pre-processed frames to the ONNX quantizer for range calibration."""

    def __init__(self, image_dir: str, imgsz: int, input_name: str, max_images: int):
        self.input_name = input_name
        self.imgsz      = imgsz
        self.images     = self._load_images(image_dir, max_images)
        self._idx       = 0
        print(f"         Loaded {len(self.images)} calibration images from '{image_dir}'")

    # ------------------------------------------------------------------

    def _load_images(self, directory: str, max_n: int):
        paths = glob.glob(f"{directory}/**/*.jpg",  recursive=True)
        paths += glob.glob(f"{directory}/**/*.png", recursive=True)
        paths = paths[:max_n]

        if not paths:
            print(f"\n  ⚠  No images found in '{directory}'.")
            print(f"     Generating {max_n} synthetic calibration frames instead.\n")
            return [self._synthetic_frame() for _ in range(max_n)]

        processed = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            processed.append(self._preprocess(img))
        return processed

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize → BGR→RGB → [0,1] → BCHW float32."""
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)            # CHW → BCHW
        return img

    def _synthetic_frame(self) -> np.ndarray:
        """Fallback: random noise frame (useful for testing pipeline)."""
        return np.random.rand(1, 3, self.imgsz, self.imgsz).astype(np.float32)

    # ------------------------------------------------------------------
    # CalibrationDataReader interface

    def get_next(self):
        if self._idx >= len(self.images):
            return None
        data = {self.input_name: self.images[self._idx]}
        self._idx += 1
        return data

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_input_name(model_path: str) -> str:
    sess = ort.InferenceSession(model_path)
    return sess.get_inputs()[0].name


def benchmark(model_path: str, imgsz: int, runs: int = 20) -> float:
    """Returns mean latency in ms over `runs` inference calls."""
    sess = ort.InferenceSession(
        model_path,
        sess_options=_opt_session(),
    )
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warm-up
    for _ in range(3):
        sess.run(None, {input_name: dummy})

    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {input_name: dummy})
    return (time.perf_counter() - t0) / runs * 1000


def _opt_session() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print("  PPE Detector — ONNX INT8 Static Quantization")
    print(f"{'='*55}\n")

    # ── Validate input ──────────────────────────────────────────────────
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            f"Run export_onnx.py first."
        )

    input_name = get_input_name(args.model)
    print(f"[1/4]  Input model  : {args.model}  ({file_size_mb(args.model):.2f} MB)")
    print(f"       Input tensor : '{input_name}'  shape [1, 3, {args.imgsz}, {args.imgsz}]")

    # ── Calibration data ────────────────────────────────────────────────
    print(f"\n[2/4]  Building calibration dataset ...")
    calib_reader = PPECalibrationReader(
        image_dir  = args.calib_images,
        imgsz      = args.imgsz,
        input_name = input_name,
        max_images = args.num_calib,
    )

    # ── Quantize ────────────────────────────────────────────────────────
    print(f"\n[3/4]  Running INT8 static quantization ...")
    quantize_static(
        model_input          = args.model,
        model_output         = args.output,
        calibration_data_reader = calib_reader,
        quant_format         = QuantFormat.QDQ,   # QDQ = standard for ORT
        activation_type      = QuantType.QInt8,
        weight_type          = QuantType.QInt8,
    )
    print(f"         ✓ Saved → {args.output}")

    # ── Validate & benchmark ─────────────────────────────────────────────
    print(f"\n[4/4]  Benchmarking FP32 vs INT8 (CPU, {20} runs each) ...")
    fp32_lat  = benchmark(args.model,  args.imgsz)
    int8_lat  = benchmark(args.output, args.imgsz)
    fp32_size = file_size_mb(args.model)
    int8_size = file_size_mb(args.output)

    speedup      = fp32_lat / int8_lat if int8_lat > 0 else 0
    size_ratio   = fp32_size / int8_size if int8_size > 0 else 0

    print(f"\n{'─'*45}")
    print(f"  {'Metric':<22} {'FP32':>8}  {'INT8':>8}  {'Gain':>8}")
    print(f"  {'─'*42}")
    print(f"  {'Latency (ms)':<22} {fp32_lat:>8.1f}  {int8_lat:>8.1f}  {speedup:>7.2f}x")
    print(f"  {'Model size (MB)':<22} {fp32_size:>8.2f}  {int8_size:>8.2f}  {size_ratio:>7.2f}x")
    print(f"{'─'*45}")
    print(f"\n  ✅ Quantization complete.")
    print(f"     Deploy '{args.output}' with serve_onnx.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
