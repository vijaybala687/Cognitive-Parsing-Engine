import json
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
import uvicorn

# ---------------------------------------------------------
# 1. Pipeline Initialization
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / 'src' / 'saved_models' / 'hybrid_eegnet_4class.onnx')
DATA_X_PATH = str(BASE_DIR / 'data' / 'processed' / 'X.npy')
DATA_Y_PATH = str(BASE_DIR / 'data' / 'processed' / 'y.npy')

print("[INIT] Booting 4-Class ONNX Engine...")
session = ort.InferenceSession(MODEL_PATH)
X_raw = np.load(DATA_X_PATH).astype(np.float32)
y_raw = np.load(DATA_Y_PATH).astype(np.int64)

if np.min(y_raw) > 0: y_raw = y_raw - np.min(y_raw)

class_pools = { i: np.where(y_raw == i)[0] for i in range(4) }
CLASS_LABELS = {0: "Left Hand", 1: "Right Hand", 2: "Both Fists (UP)", 3: "Both Feet (DOWN)"}
DIRECTION_MAP = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_inference(sample_idx: int):
    raw_epoch = X_raw[sample_idx]
    mean = np.mean(raw_epoch, axis=-1, keepdims=True)
    std = np.std(raw_epoch, axis=-1, keepdims=True) + 1e-8
    norm_epoch = (raw_epoch - mean) / std
    
    input_tensor = np.expand_dims(np.expand_dims(norm_epoch, axis=0), axis=0)
    logits = session.run(['output'], {'input': input_tensor})[0][0]
    probs = softmax(logits)
    pred_class = int(np.argmax(logits))
    
    # EXPLOIT RAW DATA: Extract true EEG channels (Approx C3, Cz, C4, Pz in PhysioNet)
    # Downsample by a factor of 4 (160 time steps) to keep browser rendering ultra-smooth
    eeg_waveforms = norm_epoch[[7, 9, 11, 30], ::4].tolist()
    
    return pred_class, float(probs[pred_class]) * 100.0, eeg_waveforms

# ---------------------------------------------------------
# 2. FastAPI Application
# ---------------------------------------------------------
app = FastAPI(title="Cognitive Parsing Engine")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws")
async def bci_stream(websocket: WebSocket):
    await websocket.accept()
    cmd_to_class = {"INJECT_LEFT": 0, "INJECT_RIGHT": 1, "INJECT_UP": 2, "INJECT_DOWN": 3}
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            command = payload.get("command")
            
            if command in cmd_to_class:
                target_class = cmd_to_class[command]
                sample_idx = int(np.random.choice(class_pools[target_class]))
                pred_class, conf, waveforms = run_inference(sample_idx)
                
                response = {
                    "injected_true": CLASS_LABELS[target_class],
                    "prediction": CLASS_LABELS[pred_class],
                    "confidence": conf,
                    "match": pred_class == target_class,
                    "move_cmd": DIRECTION_MAP[pred_class],
                    "waveforms": waveforms,
                    "target_class": target_class
                }
                await websocket.send_json(response)
            elif command == "RESET":
                await websocket.send_json({"move_cmd": "CENTER"})
                
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=True)