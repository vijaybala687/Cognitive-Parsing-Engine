# src/mapping/command_mapper.py

def translate_intent_to_action(predicted_state: str) -> str:
    """
    Maps the ML model's motor imagery classification to a directional action.
    """
    mapping_dictionary = {
        "Left-hand imagery": "LEFT",
        "Right-hand imagery": "RIGHT",
        "Both-hands imagery": "UP",
        "Both-feet imagery": "DOWN"
    }
    
    # Returns "NONE" if the state isn't recognized, acting as a fail-safe
    return mapping_dictionary.get(predicted_state, "NONE")