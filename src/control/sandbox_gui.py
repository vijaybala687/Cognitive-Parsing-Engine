# src/control/sandbox_gui.py
import pygame
import sys
import time

def run_virtual_sandbox(get_command_callback=None):
    """Launches a Pygame window and accepts commands from an external function."""
    pygame.init()
    
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Cognitive Parsing Engine - Virtual Sandbox")
    
    cursor_color = (255, 0, 0)
    cursor_radius = 15
    cursor_pos = [width // 2, height // 2]
    move_step = 5 # Speed of the cursor
    
    clock = pygame.time.Clock()
    running = True
    
    # Track time to fetch new commands every 2 seconds
    last_command_time = time.time()
    current_command = "NONE"
    
    print("Sandbox running. Awaiting directional commands...")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # If a callback is provided, fetch a new command every 2 seconds
        if get_command_callback and (time.time() - last_command_time > 2.0):
            current_command = get_command_callback()
            last_command_time = time.time()
        
        # Apply continuous movement based on the active command
        if current_command == "UP":
            cursor_pos[1] -= move_step
        elif current_command == "DOWN":
            cursor_pos[1] += move_step
        elif current_command == "LEFT":
            cursor_pos[0] -= move_step
        elif current_command == "RIGHT":
            cursor_pos[0] += move_step
            
        # Keep cursor inside the screen bounds
        cursor_pos[0] = max(cursor_radius, min(width - cursor_radius, cursor_pos[0]))
        cursor_pos[1] = max(cursor_radius, min(height - cursor_radius, cursor_pos[1]))
        
        screen.fill((30, 30, 30)) 
        pygame.draw.circle(screen, cursor_color, cursor_pos, cursor_radius)
        pygame.display.flip()
        
        clock.tick(30) 
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_virtual_sandbox()