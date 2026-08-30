const DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/bci-stream";

export function createBCIStream({ onOpen, onMessage, onError, onClose } = {}) {
  const socketUrl = import.meta.env.VITE_BCI_WS_URL || DEFAULT_WS_URL;
  const socket = new WebSocket(socketUrl);

  socket.onopen = () => {
    if (onOpen) onOpen();
  };

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (onMessage) onMessage(payload);
  };

  socket.onerror = (event) => {
    if (onError) onError(event);
  };

  socket.onclose = () => {
    if (onClose) onClose();
  };

  return socket;
}

export async function getPrediction() {
  return { movement: "CENTER", confidence: 0 };
}