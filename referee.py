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


# backend/referee.py (sigue)
def bbox_center(bb): 
    x1,y1,x2,y2 = bb; return ((x1+x2)/2.0, (y1+y2)/2.0)

def detect_events(jsonl_path, video_path, H=None):
    pitch_poly, left_goal, right_goal = field_polygons()
    events = []
    last_ball = None; last_ball_in_pitch = True
    last_players = {}
    v = cv2.VideoCapture(video_path); fps = v.get(cv2.CAP_PROP_FPS) or 25
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line); frame = rec["frame"]; t = frame/ fps
            ball = None; players=[]
            for d in rec["dets"]:
                if d["cls"]==32: ball = d
                if d["cls"]==0:  players.append(d)
            # BALL LOGIC
            if ball:
                cx,cy = bbox_center(ball["xyxy"])
                if H is not None:
                    fx,fy = pixel_to_field(H, cx, cy)
                    pt = geom.Point(fx,fy)
                    in_pitch = pitch_poly.contains(pt)
                    into_goal = left_goal.contains(pt) or right_goal.contains(pt)
                else:
                    in_pitch = True
                    into_goal = False
                # GOAL
                if into_goal and last_ball and last_ball.get("in_pitch",True):
                    events.append({"type":"goal", "time": t})
                # OUT
                if (last_ball_in_pitch and not in_pitch):
                    events.append({"type":"ball_out", "time": t})
                last_ball_in_pitch = in_pitch
                last_ball = {"cx":cx,"cy":cy,"in_pitch":in_pitch}
            # COLLISION/CAÍDA (heurística)
            for p in players:
                pid = p["id"]
                cx,cy = bbox_center(p["xyxy"])
                h = abs(p["xyxy"][3]-p["xyxy"][1])
                if pid in last_players:
                    vx = cx - last_players[pid]["cx"]; vy = cy - last_players[pid]["cy"]
                    vmag = math.hypot(vx,vy)
                    # caída: altura de caja ↑ (sube visualmente por foreshortening) o ↓ brusco + v baja
                    dh = h - last_players[pid]["h"]
                    if vmag < 0.5 and dh > 15:  # umbrales empíricos para demo
                        events.append({"type":"collision_fall", "time": t, "player_id": int(pid)})
                last_players[pid] = {"cx":cx,"cy":cy,"h":h}
    return dedup(events)

def dedup(events, tol=1.0):
    out=[]; last={}
    for e in events:
        key=e["type"]; t=e["time"]
        if key not in last or abs(t-last[key])>tol:
            out.append(e); last[key]=t
    return out

