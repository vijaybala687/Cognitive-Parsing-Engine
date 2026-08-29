import "./PredictionPanel.css";

function PredictionPanel({ prediction = "LEFT", confidence = 98 }) {

  const handleInject = (direction) => {
    console.log("Injected:", direction);

    // Your friend will replace this with:
    // fetch("/predict", { method: "POST", body: JSON.stringify({ intent: direction }) })
  };

  return (
    <aside className="predictionPanel">

      <h2>Live Prediction</h2>

      <div className="prediction-card">
        <span>Prediction</span>
        <h1>{prediction}</h1>
      </div>

      <div className="prediction-card">
        <span>Confidence</span>
        <h1>{confidence}%</h1>
      </div>

      <div className="prediction-card">
        <span>Status</span>
        <h1>ACTIVE</h1>
      </div>

      <div className="prediction-card">
        <span>Model</span>
        <h1>CSP + LDA</h1>
      </div>

      <div className="testing-panel">

        <h3>Model Testing</h3>

        <div className="button-grid">

          <button onClick={() => handleInject("LEFT")}>
            LEFT
          </button>

          <button onClick={() => handleInject("RIGHT")}>
            RIGHT
          </button>

          <button onClick={() => handleInject("UP")}>
            UP
          </button>

          <button onClick={() => handleInject("DOWN")}>
            DOWN
          </button>

          <button
            className="reset-btn"
            onClick={() => handleInject("RESET")}
          >
            RESET
          </button>

        </div>

      </div>

    </aside>
  );
}

export default PredictionPanel;