import { useState, useEffect } from "react";
import "./CursorArena.css";
import { getFakePrediction } from "../../services/eegService";

function CursorArena() {
  // Cursor Position
  const [position, setPosition] = useState({
    x: 0,
    y: 0,
  });

  // Current Prediction
  const [prediction, setPrediction] = useState("CENTER");

  // Confidence
  const [confidence, setConfidence] = useState(98);

  useEffect(() => {
    const interval = setInterval(() => {
      const data = getFakePrediction();

      setPrediction(data.movement);
      setConfidence(data.confidence);

      switch (data.movement) {
        case "LEFT":
          setPosition({ x: -120, y: 0 });
          break;

        case "RIGHT":
          setPosition({ x: 120, y: 0 });
          break;

        case "UP":
          setPosition({ x: 0, y: -120 });
          break;

        case "DOWN":
          setPosition({ x: 0, y: 120 });
          break;

        default:
          setPosition({ x: 0, y: 0 });
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="arena-wrapper">
      <div className="arena-header">
        <h2>Cursor Control Arena</h2>

        <div className="prediction-pill">
          Live Prediction
        </div>
      </div>

      <div className="arena">
        {/* Center Cross */}
        <div className="cross horizontal"></div>
        <div className="cross vertical"></div>

        {/* Cursor */}
        <div
          className="cursor"
          style={{
            transform: `translate(calc(-50% + ${position.x}px), calc(-50% + ${position.y}px))`,
          }}
        >
          <div className="cursor-core"></div>
          <div className="cursor-ring"></div>
          <div className="cursor-glow"></div>
        </div>
      </div>

      <div className="arena-footer">
        <div>
          <p>Current Movement</p>
          <h3>{prediction}</h3>
        </div>

        <div>
          <p>Confidence</p>
          <h3>{confidence}%</h3>
        </div>
      </div>
    </section>
  );
}

export default CursorArena;