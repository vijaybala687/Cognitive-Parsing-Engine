import { useEffect, useState } from "react";
import "./CursorArena.css";

const EASING_FACTOR = 0.2;

function CursorArena({ prediction = "CENTER", confidence = 0, targetPos = [320, 280] }) {
  const [position, setPosition] = useState([320, 280]);

  useEffect(() => {
    let animationFrameId;
    let currentPos = [...position];

    const animate = () => {
      // Smooth interpolation towards target position
      currentPos[0] += (targetPos[0] - currentPos[0]) * EASING_FACTOR;
      currentPos[1] += (targetPos[1] - currentPos[1]) * EASING_FACTOR;

      setPosition([...currentPos]);

      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrameId);
  }, [targetPos]);

  return (
    <section className="arena-wrapper">
      <div className="arena-header">
        <h2>Cursor Control Arena</h2>

        <div className="prediction-pill">Live Prediction</div>
      </div>

      <div className="arena">
        <div className="cross horizontal"></div>
        <div className="cross vertical"></div>

        <div
          className="cursor"
          style={{
            left: `${position[0]}px`,
            top: `${position[1]}px`,
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