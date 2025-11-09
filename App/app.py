# app.py
import os, io, json, tempfile, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import gradio as gr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


CLASS_LABELS = [
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Clearance",
    "Shots on target",
    "Shots off target",
    "Corner",
]
LABEL2ID = {c:i for i,c in enumerate(CLASS_LABELS)}


T = 16               # frames por clip
SIDE_SHORT = 256     # resize por lado corto
H = W = 112          # crop final
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


NUM_CLASSES = len(CLASS_LABELS)
CKPT_PATH = "modelo.pth"  # e.g. "/kaggle/working/modelo.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return n, float(fps)

def center_crop_112(img):
    h0, w0 = img.shape[:2]
    y0 = max(0, (h0 - H)//2)
    x0 = max(0, (w0 - W)//2)
    return img[y0:y0+H, x0:x0+W, :]

def resize_by_short_side(img, side_short=SIDE_SHORT):
    h0, w0 = img.shape[:2]
    if h0 < w0:
        new_h = side_short
        new_w = int(round(w0 * (side_short / h0)))
    else:
        new_w = side_short
        new_h = int(round(h0 * (side_short / w0)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def sample_indices_uniform(total_frames, T):
    # indices uniformes en [0, total_frames)
    if total_frames <= 0:
        return None
    if total_frames < T:
        # repetimos lo mínimo posible, pero garantizamos T
        xs = np.linspace(0, total_frames-1, num=T)
        return np.floor(xs).astype(int).tolist()
    xs = np.linspace(0, total_frames-1, num=T, endpoint=False)
    return np.floor(xs).astype(int).tolist()

def video_to_tensor(path):
    n, fps = probe_video(path)
    idxs = sample_indices_uniform(n, T)
    if idxs is None:
        raise RuntimeError("No se pudo muestrear frames del video.")
    cap = cv2.VideoCapture(path)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Fallo leyendo frame {idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_by_short_side(frame, SIDE_SHORT)
        frame = center_crop_112(frame)
        frames.append(frame)
    cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
    arr = (arr - MEAN) / (STD + 1e-6)
    arr = np.transpose(arr, (3,0,1,2))  # -> [3,T,H,W]
    ten = torch.from_numpy(arr).unsqueeze(0)  # [1,3,T,H,W]
    return ten


def build_model(num_classes=NUM_CLASSES):
    arch = os.environ.get("ARCH", "r3d_18").lower()  # r3d_18 | r2plus1d_18 | mc3_18 | s3d, etc.
    if arch == "r2plus1d_18":
        model = torchvision.models.video.r2plus1d_18(weights=None)
    elif arch == "mc3_18":
        model = torchvision.models.video.mc3_18(weights=None)
    else:
        model = torchvision.models.video.r3d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


_model = build_model(NUM_CLASSES).to(DEVICE)
_model.eval()

def try_load_ckpt(model, ckpt_path, ignore_prefixes=("stem.", "fc.")):
    if not ckpt_path or not Path(ckpt_path).exists():
        return "Sin checkpoint (usando pesos aleatorios)."

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    sd = ckpt.get("state_dict", ckpt)
    # quitar posibles prefijos 'module.' o 'model.'
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}

    model_sd = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in sd.items():
        # ignora capas conflictivas por nombre
        if any(k.startswith(p) for p in ignore_prefixes):
            skipped.append((k, "ignored_by_prefix"))
            continue
        # copia sólo si la clave existe y la forma coincide
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append((k, "shape_mismatch"))

    msg = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    miss = len(msg.missing_keys)
    unexp = len(msg.unexpected_keys)
    return (f"Cargado parcialmente: {loaded} capas. "
            f"missing={miss}, unexpected={unexp}, skipped={len(skipped)} "
            f"(stem/fc o mismatch).")

LOAD_MSG = try_load_ckpt(_model, CKPT_PATH)


@torch.inference_mode()
def predict(video_path, topk=3):
    x = video_to_tensor(video_path).to(DEVICE)  # [1,3,T,H,W]
    logits = _model(x)                          # [1,C]
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = probs.argmax().item()
    top_label = CLASS_LABELS[top_idx]
    # top-k
    k = min(topk, len(CLASS_LABELS))
    topk_idx = probs.argsort()[::-1][:k].tolist()
    topk_list = [(CLASS_LABELS[i], float(probs[i])) for i in topk_idx]
    return top_label, probs.tolist(), topk_list

def make_result_py(pred_label, topk_list):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Auto-generado por la app de Gradio",
        f"# Fecha: {now}",
        "",
        "def main():",
        f"    pred_label = {repr(pred_label)}",
        f"    topk = {repr([(lbl, round(score,6)) for lbl, score in topk_list])}",
        '    print(f"Predicted class: {pred_label}")',
        '    print("Top-k:")',
        '    for lbl, sc in topk:',
        '        print(f"  - {lbl}: {sc:.6f}")',
        "",
        'if __name__ == "__main__":',
        "    main()",
        ""
    ]
    content = "\n".join(lines)
    tmpdir = tempfile.mkdtemp(prefix="pred_py_")
    out_path = os.path.join(tmpdir, "prediction_result.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


EXPLAIN = f"""
**Modelo:** r3d_18 (torchvision) → {NUM_CLASSES} clases  
**Checkpoint:** {'ninguno' if not CKPT_PATH else CKPT_PATH}  
**Carga:** {LOAD_MSG}
  
**Preprocesamiento:**  
- T = {T} frames uniformes en el clip subido  
- Resize por lado corto = {SIDE_SHORT}, crop centrado = {H}×{W}  
- Normalización mean=0.45, std=0.225 (Kinetics-like)  
"""

def infer_and_build_py(video_file):
    if video_file is None:
        return "Sube un video.", None, None
    path = video_file if isinstance(video_file, str) else video_file.name
    try:
        pred_label, probs, topk_list = predict(path, topk=3)
        # salida textual amigable
        txt = f"Predicción: **{pred_label}**\n\nTop-3:\n" + "\n".join(
            [f"- {lbl}: {score:.4f}" for lbl, score in topk_list]
        )
        
        py_path = make_result_py(pred_label, topk_list)
        # JSON con todas las probabilidades (por si lo quieres)
        probs_json = {CLASS_LABELS[i]: float(p) for i, p in enumerate(probs)}
        return txt, py_path, json.dumps(probs_json, indent=2)
    except Exception as e:
        return f"Error: {e}", None, None

with gr.Blocks(title="Soccer Event Classifier") as demo:
    gr.Markdown("# ⚽ Soccer Event Classifier\nSube un clip y obtén la clase + un `.py` con el resultado.")
    gr.Markdown(EXPLAIN)

    with gr.Row():
        inp = gr.Video(label="Clip de entrada (mp4/mkv/avi)")
    with gr.Row():
        btn = gr.Button("Clasificar", variant="primary")
    with gr.Row():
        out_txt = gr.Markdown(label="Resultado")
    with gr.Row():
        out_file = gr.File(label="Descargar .py con la predicción", file_count="single")
    with gr.Row():
        out_json = gr.JSON(label="Probabilidades por clase (JSON)")

    btn.click(fn=infer_and_build_py, inputs=[inp], outputs=[out_txt, out_file, out_json])

if __name__ == "__main__":
    # Para correr local: python app.py
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
