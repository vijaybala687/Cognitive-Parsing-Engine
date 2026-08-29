import { useState, useEffect } from "react";

import Header from "../../components/Header/Header";
import Sidebar from "../../components/Sidebar/Sidebar";
import CursorArena from "../../components/CursorArena/CursorArena";
import PredictionPanel from "../../components/PredictionPanel/PredictionPanel";
import EEGWave from "../../components/EEGWave/EEGWave";
import Footer from "../../components/Footer/Footer";

import { getPrediction } from "../../services/api";

import "./Dashboard.css";

function Dashboard() {

  const [prediction, setPrediction] = useState("CENTER");
  const [confidence, setConfidence] = useState(98);

  useEffect(() => {

    const fetchPrediction = async () => {

      try {

        const data = await getPrediction();

        setPrediction(data.movement);
        setConfidence(data.confidence);

      } catch (error) {

        console.error("Error fetching prediction:", error);

      }

    };

    // First fetch immediately
    fetchPrediction();

    // Then fetch every second
    const interval = setInterval(fetchPrediction, 1000);

    return () => clearInterval(interval);

  }, []);

  return (
    <>
      <Header />

      <main className="main-layout">

        <Sidebar />

        <CursorArena
          prediction={prediction}
          confidence={confidence}
        />

        <PredictionPanel
          prediction={prediction}
          confidence={confidence}
        />

      </main>

      <EEGWave />

      <Footer />
    </>
  );
}

export default Dashboard;