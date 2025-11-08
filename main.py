
# backend/main.py
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from referee import run_tracking, detect_events
import os, uuid, json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STORE = "runs"; os.makedirs(STORE, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile):
    vid_path = os.path.join(STORE, f"{uuid.uuid4()}.mp4")
    with open(vid_path, "wb") as f: f.write(await file.read())
    det_path = vid_path.replace(".mp4",".jsonl")
    run_tracking(vid_path, det_path)
    events = detect_events(det_path, vid_path, H=None)  # si logras H, pásala aquí
    # guarda para reproducir en frontend
    with open(vid_path.replace(".mp4",".events.json"),"w") as f: json.dump(events,f)
    return {"video": os.path.basename(vid_path), "events": events}
