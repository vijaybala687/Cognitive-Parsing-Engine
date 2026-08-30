import { useEffect, useRef, useState } from "react";

import Header from "../../components/Header/Header";
import Sidebar from "../../components/Sidebar/Sidebar";
import CursorArena from "../../components/CursorArena/CursorArena";
import PredictionPanel from "../../components/PredictionPanel/PredictionPanel";
import EEGWave from "../../components/EEGWave/EEGWave";
import Footer from "../../components/Footer/Footer";

import { createBCIStream } from "../../services/api";

import "./Dashboard.css";

const STEP_SIZE = 50;
const ARENA_WIDTH = 640;
const ARENA_HEIGHT = 560;
const CENTER_X = ARENA_WIDTH / 2;
const CENTER_Y = ARENA_HEIGHT / 2;
const MIN_X = 0;
const MAX_X = ARENA_WIDTH;
const MIN_Y = 0;
const MAX_Y = ARENA_HEIGHT;

function Dashboard() {
  const [prediction, setPrediction] = useState("CENTER");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("CONNECTING");
  const [targetPos, setTargetPos] = useState([CENTER_X, CENTER_Y]);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = createBCIStream({
      onOpen: () => setStatus("ACTIVE"),
      onMessage: (payload) => {
        const intent = payload.intent || "CENTER";
        const nextConfidence = Math.round((Number(payload.confidence) || 0) * 100);

        setPrediction(intent);
        setConfidence(nextConfidence);
        setStatus("LIVE");

        // Update target position based on prediction
        setTargetPos((prev) => {
          let newX = prev[0];
          let newY = prev[1];

          if (intent === "LEFT") newX -= STEP_SIZE;
          else if (intent === "RIGHT") newX += STEP_SIZE;
          else if (intent === "UP") newY -= STEP_SIZE;
          else if (intent === "DOWN") newY += STEP_SIZE;

          // Clamp to arena bounds
          newX = Math.max(MIN_X, Math.min(MAX_X, newX));
          newY = Math.max(MIN_Y, Math.min(MAX_Y, newY));

          return [newX, newY];
        });
      },
      onError: () => setStatus("ERROR"),
      onClose: () => setStatus("DISCONNECTED"),
    });

    socketRef.current = socket;

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const sendTrigger = (direction) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      setStatus("CONNECTING");
      return;
    }

    socketRef.current.send(
      JSON.stringify({
        type: "trigger",
        intent: direction,
      })
    );
  };

  const resetCursor = () => {
    setTargetPos([CENTER_X, CENTER_Y]);
  };

  return (
    <>
      <Header />

      <main className="main-layout">
        <Sidebar />

        <CursorArena
          prediction={prediction}
          confidence={confidence}
          targetPos={targetPos}
        />

        <PredictionPanel
          prediction={prediction}
          confidence={confidence}
          status={status}
          onSendTrigger={sendTrigger}
          onReset={resetCursor}
        />
      </main>

      <EEGWave />

      <Footer />
    </>
  );
}

export default Dashboard;