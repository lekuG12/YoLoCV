"""
tests/test_pipeline.py
-----------------------
Basic sanity tests for the export → quantize → serve pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# export_onnx.py
# ---------------------------------------------------------------------------

def test_export_args_defaults():
    from export_onnx import parse_args
    sys.argv = ["export_onnx.py"]
    args = parse_args()
    assert args.imgsz == 640
    assert args.batch == 1
    assert args.opset == 17
    assert args.output == "ppe_detector.onnx"


# ---------------------------------------------------------------------------
# quantize_onnx.py — calibration reader
# ---------------------------------------------------------------------------

def test_calibration_reader_synthetic_fallback(tmp_path):
    from quantize_onnx import PPECalibrationReader

    empty_dir = tmp_path / "no_images_here"
    empty_dir.mkdir()

    reader = PPECalibrationReader(
        image_dir=str(empty_dir),
        imgsz=640,
        input_name="images",
        max_images=5,
    )
    assert len(reader.images) == 5

    batch = reader.get_next()
    assert "images" in batch
    assert batch["images"].shape == (1, 3, 640, 640)
    assert batch["images"].dtype == np.float32


def test_calibration_reader_rewind(tmp_path):
    from quantize_onnx import PPECalibrationReader

    empty_dir = tmp_path / "no_images"
    empty_dir.mkdir()

    reader = PPECalibrationReader(
        image_dir=str(empty_dir),
        imgsz=320,
        input_name="images",
        max_images=3,
    )

    # Exhaust the reader
    while reader.get_next() is not None:
        pass
    assert reader.get_next() is None

    # Rewind should reset iteration
    reader.rewind()
    assert reader.get_next() is not None


# ---------------------------------------------------------------------------
# serve_onnx.py — postprocessing logic
# ---------------------------------------------------------------------------

def test_ppe_classes_defined():
    from serve_onnx import PPE_CLASSES
    assert len(PPE_CLASSES) == 6
    assert "helmet" in PPE_CLASSES
    assert "no-mask" in PPE_CLASSES


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from serve_onnx import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# File structure sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "export_onnx.py",
    "quantize_onnx.py",
    "serve_onnx.py",
    "requirements.txt",
    "README.md",
])
def test_required_files_exist(filename):
    root = Path(__file__).resolve().parent.parent
    assert (root / filename).exists(), f"Missing required file: {filename}"
