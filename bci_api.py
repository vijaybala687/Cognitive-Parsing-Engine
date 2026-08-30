import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "src" / "saved_models" / "hybrid_eegnet_4class.onnx"
DATA_PATH = BASE_DIR / "data" / "processed" / "X.npy"
LABEL_PATH = BASE_DIR / "data" / "processed" / "y.npy"

CLASS_LABELS = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
}

INTENT_TO_CLASS = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}

app = FastAPI(title="Cognitive Parsing Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_session = None
X_samples = None
y_labels = None


@app.on_event("startup")
def startup_event():
    global model_session, X_samples, y_labels

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model_session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    X_samples = np.load(str(DATA_PATH)).astype(np.float32)
    y_labels = np.load(str(LABEL_PATH)).astype(np.int64)


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    mean = np.mean(epoch, axis=-1, keepdims=True)
    std = np.std(epoch, axis=-1, keepdims=True) + 1e-8
    return (epoch - mean) / std


def infer_with_model(epoch: np.ndarray):
    if model_session is None:
        raise RuntimeError("Model session has not been initialized")

    norm_epoch = normalize_epoch(epoch)
    input_tensor = np.expand_dims(np.expand_dims(norm_epoch, axis=0), axis=0)
    output_name = model_session.get_outputs()[0].name
    logits = model_session.run([output_name], {"input": input_tensor})[0][0]

    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / probabilities.sum()
    pred_index = int(np.argmax(logits))
    pred_label = CLASS_LABELS.get(pred_index, "CENTER")
    confidence = float(probabilities[pred_index])

    return pred_label, confidence


def select_sample_for_intent(intent: str):
    if X_samples is None or len(X_samples) == 0:
        raise RuntimeError("No EEG samples available for inference")

    target_class = INTENT_TO_CLASS.get(intent.upper())
    if target_class is not None:
        class_indices = np.where(y_labels == target_class)[0]
        if len(class_indices) > 0:
            sample_index = int(np.random.choice(class_indices))
            return X_samples[sample_index]

    sample_index = int(np.random.randint(0, len(X_samples)))
    return X_samples[sample_index]


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_session is not None}


@app.websocket("/ws/bci-stream")
async def bci_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "intent": "CENTER",
                    "confidence": 0.0,
                    "timestamp": int(time.time() * 1000),
                    "error": "invalid_json",
                })
                continue

            if message.get("type") == "trigger":
                intent = str(message.get("intent", "LEFT")).upper()
                sample = select_sample_for_intent(intent)
                predicted_label, confidence = infer_with_model(sample)

                payload = {
                    "intent": predicted_label,
                    "confidence": round(float(confidence), 4),
                    "timestamp": int(time.time() * 1000),
                }
                await websocket.send_json(payload)

            elif message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": int(time.time() * 1000),
                })

            else:
                await websocket.send_json({
                    "intent": "CENTER",
                    "confidence": 0.0,
                    "timestamp": int(time.time() * 1000),
                    "error": "unsupported_message_type",
                })

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bci_api:app", host="0.0.0.0", port=8000, reload=False)
