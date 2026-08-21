# src/control/cursor_controller.py
import pyautogui

# FAILSAFE: Moving the physical mouse to any corner of the screen will abort the program.
pyautogui.FAILSAFE = True 

def move_physical_cursor(direction: str, step_pixels: int = 50):
    """
    Moves the OS-level mouse cursor based on the directional command.
    """
    try:
        if direction == "UP":
            pyautogui.move(0, -step_pixels, duration=0.1)
        elif direction == "DOWN":
            pyautogui.move(0, step_pixels, duration=0.1)
        elif direction == "LEFT":
            pyautogui.move(-step_pixels, 0, duration=0.1)
        elif direction == "RIGHT":
            pyautogui.move(step_pixels, 0, duration=0.1)
        else:
            pass # No movement for "NONE" or unrecognized commands
            
    except pyautogui.FailSafeException:
        print("CRITICAL: PyAutoGUI Fail-Safe triggered. Cursor control aborted.")