# src/control/sandbox_gui.py
import pygame
import sys

def run_virtual_sandbox():
    """Launches a Pygame window to safely visualize BCI cursor movements."""
    pygame.init()
    
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Cognitive Parsing Engine - Virtual Sandbox")
    
    # Cursor properties
    cursor_color = (255, 0, 0)
    cursor_radius = 15
    cursor_pos = [width // 2, height // 2]
    move_step = 20
    
    clock = pygame.time.Clock()
    running = True
    
    print("Sandbox running. Awaiting directional commands...")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # In a full integration, this command would come from the ML model pipeline
        # For testing, we are simulating a "NONE" command
        simulated_command = "NONE" 
        
        if simulated_command == "UP":
            cursor_pos[1] -= move_step
        elif simulated_command == "DOWN":
            cursor_pos[1] += move_step
        elif simulated_command == "LEFT":
            cursor_pos[0] -= move_step
        elif simulated_command == "RIGHT":
            cursor_pos[0] += move_step
            
        # Keep cursor inside the screen bounds
        cursor_pos[0] = max(cursor_radius, min(width - cursor_radius, cursor_pos[0]))
        cursor_pos[1] = max(cursor_radius, min(height - cursor_radius, cursor_pos[1]))
        
        screen.fill((30, 30, 30)) # Dark gray background
        pygame.draw.circle(screen, cursor_color, cursor_pos, cursor_radius)
        pygame.display.flip()
        
        clock.tick(30) # 30 FPS refresh rate
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_virtual_sandbox()