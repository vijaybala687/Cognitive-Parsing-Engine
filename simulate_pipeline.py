# simulate_pipeline.py
import random
from src.mapping.command_mapper import translate_intent_to_action
from src.control.sandbox_gui import run_virtual_sandbox

# The four motor imagery classes expected from the final model, plus a resting state
MOCK_ML_PREDICTIONS = [
    "Left-hand imagery",
    "Right-hand imagery",
    "Both-hands imagery",
    "Both-feet imagery",
    "Resting state" 
]

def mock_ml_inference():
    """
    Simulates the ML pipeline outputting a prediction.
    It passes the raw string through your command mapper to get the action.
    """
    raw_prediction = random.choice(MOCK_ML_PREDICTIONS)
    print(f"\n[Brainwave Detected] Model predicted: {raw_prediction}")
    
    # Your translation layer goes to work here
    action = translate_intent_to_action(raw_prediction)
    print(f"[Command Mapper] Translated to action: {action}")
    
    return action

if __name__ == "__main__":
    print("Initializing Simulation Bridge...")
    # Launch the sandbox and pass it the mock ML function
    run_virtual_sandbox(get_command_callback=mock_ml_inference)