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

# backend/referee.py (sigue)
import shapely.geometry as geom

PITCH_W, PITCH_H = 105.0, 68.0  # metros
# definimos polígonos en coordenadas de campo: áreas de gol (detrás de la línea)
GOAL_ZONE_DEPTH = 2.5  # m detrás de la línea de gol

def estimate_homography(frame_bgr):
    # BOILERPLATE: segmentar césped y detectar líneas blancas (sobresimplificado)
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=200, maxLineGap=20)
    # En demo: usa 4 esquinas del rectángulo visible (heurística) o pide 4 clics si falla
    # Aquí devolvemos None si no logramos algo estable en la primera llamada.
    return None, None

def field_polygons():
    # zonas detrás de las líneas de gol, para marcar “entra a portería”
    left_goal = geom.Polygon([(0,-5),(0, PITCH_H+5),(GOAL_ZONE_DEPTH, PITCH_H+5),(GOAL_ZONE_DEPTH,-5)])
    right_goal = geom.Polygon([(PITCH_W-GOAL_ZONE_DEPTH,-5),(PITCH_W, -5),(PITCH_W, PITCH_H+5),(PITCH_W-GOAL_ZONE_DEPTH, PITCH_H+5)])
    pitch_poly = geom.Polygon([(0,0),(PITCH_W,0),(PITCH_W,PITCH_H),(0,PITCH_H)])
    return pitch_poly, left_goal, right_goal

def pixel_to_field(H, px, py):
    pt = np.array([px, py, 1.0]); w = H @ pt
    return (w[0]/w[2], w[1]/w[2])
