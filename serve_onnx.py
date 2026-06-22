"""
serve_onnx.py
-------------
Lightweight FastAPI inference server for the quantized PPE detector.
Accepts image uploads and returns bounding boxes + class labels.

Usage:
    pip install fastapi uvicorn onnxruntime opencv-python numpy pillow
    python serve_onnx.py --model ppe_detector_int8.onnx --port 8000

    # Test:
    curl -X POST http://localhost:8000/detect \
         -F "file=@test_image.jpg"
"""

import argparse
import io
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PPE_CLASSES = [
    "helmet", "no-helmet",
    "vest",   "no-vest",
    "mask",   "no-mask",
]

CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.45


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class PPEDetector:
    def __init__(self, model_path: str, imgsz: int = 640):
        self.imgsz = imgsz

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session    = ort.InferenceSession(model_path, opts)
        self.input_name = self.session.get_inputs()[0].name

        # Warm-up
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print(f"  ✓ Model loaded and warmed up: {model_path}")

    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def postprocess(
        self,
        outputs: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> List[dict]:
        """Parse YOLOv8 output tensor → list of detection dicts."""
        # YOLOv8 output shape: [1, num_classes+4, num_anchors]
        preds = outputs[0][0]          # (num_classes+4, anchors)
        preds = preds.T                # (anchors, num_classes+4)

        boxes   = preds[:, :4]
        scores  = preds[:, 4:]

        class_ids   = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= CONF_THRESHOLD
        boxes, confidences, class_ids = (
            boxes[mask], confidences[mask], class_ids[mask]
        )

        # cx, cy, w, h → x1, y1, x2, y2  (relative to input imgsz)
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) / self.imgsz * orig_w
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) / self.imgsz * orig_h
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) / self.imgsz * orig_w
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) / self.imgsz * orig_h

        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes      = xyxy.tolist(),
            scores      = confidences.tolist(),
            score_threshold = CONF_THRESHOLD,
            nms_threshold   = IOU_THRESHOLD,
        )

        results = []
        for i in (indices.flatten() if len(indices) else []):
            cls_name = PPE_CLASSES[class_ids[i]] if class_ids[i] < len(PPE_CLASSES) else str(class_ids[i])
            results.append({
                "class":      cls_name,
                "confidence": round(float(confidences[i]), 4),
                "bbox": {
                    "x1": round(float(xyxy[i, 0])),
                    "y1": round(float(xyxy[i, 1])),
                    "x2": round(float(xyxy[i, 2])),
                    "y2": round(float(xyxy[i, 3])),
                },
            })
        return results

    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray):
        orig_h, orig_w = image.shape[:2]
        inp    = self.preprocess(image)

        t0      = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: inp})
        latency = (time.perf_counter() - t0) * 1000

        detections = self.postprocess(outputs, orig_h, orig_w)
        return detections, round(latency, 2)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app     = FastAPI(title="PPE Detector — ONNX Server", version="1.0.0")
detector: PPEDetector = None   # Initialised in __main__


@app.get("/health")
def health():
    return {"status": "ok", "model": "ppe_detector_int8.onnx"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    cv_img   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    detections, latency_ms = detector.detect(cv_img)

    return JSONResponse({
        "filename":       file.filename,
        "latency_ms":     latency_ms,
        "num_detections": len(detections),
        "detections":     detections,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  type=str, default="ppe_detector_int8.onnx")
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--host",   type=str, default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    detector = PPEDetector(args.model, args.imgsz)
    print(f"\n  🚀 Serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
