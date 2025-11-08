# backend/referee.py (fragmento)
from ultralytics import YOLO
import cv2, json, numpy as np, math, os

def run_tracking(video_path, out_jsonl):
    model = YOLO("yolov8x.pt")  # si va muy justo: "yolov8n.pt"
    # genera tracks con ByteTrack por defecto
    results = model.track(source=video_path, stream=True, persist=True, verbose=False)
    fidx = 0
    with open(out_jsonl, "w") as f:
        for r in results:
            # r.boxes.id, r.boxes.cls, r.boxes.xyxy
            dets = []
            if r.boxes is not None:
                ids = (r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([])).astype(int)
                clses = r.boxes.cls.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                for i,(c,bb) in enumerate(zip(clses, xyxy)):
                    dets.append({
                        "id": int(ids[i]) if len(ids)>i else -1,
                        "cls": int(c), "xyxy": bb.tolist()
                    })
            f.write(json.dumps({"frame": fidx, "dets": dets})+"\n")
            fidx += 1
