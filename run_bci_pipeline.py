import sys
from pathlib import Path
import numpy as np
import pygame
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / 'src' / 'saved_models' / 'hybrid_eegnet_4class.onnx')
DATA_X_PATH = str(BASE_DIR / 'data' / 'processed' / 'X.npy')
DATA_Y_PATH = str(BASE_DIR / 'data' / 'processed' / 'y.npy')

print("[INIT] Loading 4-Class ONNX Engine...")
session = ort.InferenceSession(MODEL_PATH)
X_raw = np.load(DATA_X_PATH).astype(np.float32)
y_raw = np.load(DATA_Y_PATH).astype(np.int64)

idx_left = np.where(y_raw == 0)[0]
idx_right = np.where(y_raw == 1)[0]
idx_up = np.where(y_raw == 2)[0]
idx_down = np.where(y_raw == 3)[0]

CLASS_LABELS = {0: "Left Hand", 1: "Right Hand", 2: "Both Fists (UP)", 3: "Both Feet (DOWN)"}

def run_inference(sample_idx):
    raw_epoch = X_raw[sample_idx]
    mean = np.mean(raw_epoch, axis=-1, keepdims=True)
    std = np.std(raw_epoch, axis=-1, keepdims=True) + 1e-8
    norm_epoch = (raw_epoch - mean) / std
    input_tensor = np.expand_dims(np.expand_dims(norm_epoch, axis=0), axis=0)
    
    logits = session.run(['output'], {'input': input_tensor})[0][0]
    e_x = np.exp(logits - np.max(logits))
    probs = e_x / e_x.sum()
    pred_class = int(np.argmax(logits))
    return pred_class, float(probs[pred_class]) * 100.0

pygame.init()
screen = pygame.display.set_mode((1000, 700))
pygame.display.set_caption("BCI 4-Way Cursor Sandbox")
clock = pygame.time.Clock()

font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
font_body = pygame.font.SysFont("Segoe UI", 14)
font_bold = pygame.font.SysFont("Segoe UI", 14, bold=True)

cursor_pos, target_pos = [680.0, 360.0], [680.0, 360.0]
STEP_SIZE = 50.0

class Button:
    def __init__(self, y, text, cls_id):
        self.rect = pygame.Rect(40, y, 260, 36)
        self.text = text
        self.cls_id = cls_id

    def draw(self):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = (29, 78, 216) if hover else (37, 99, 235)
        if self.cls_id == -1: color = (51, 65, 85) if hover else (71, 85, 105)
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        txt = font_bold.render(self.text, True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

buttons = [
    Button(400, "Inject Left Intent (Class 0)", 0),
    Button(445, "Inject Right Intent (Class 1)", 1),
    Button(490, "Inject Up Intent (Class 2)", 2),
    Button(535, "Inject Down Intent (Class 3)", 3),
    Button(590, "Reset Cursor Position", -1)
]

last_injected, last_prediction, last_status, last_conf = "--", "--", "--", 0.0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for btn in buttons:
                if btn.rect.collidepoint(event.pos):
                    if btn.cls_id == -1:
                        target_pos = [680.0, 360.0]
                        cursor_pos = [680.0, 360.0]
                    else:
                        pool = {0: idx_left, 1: idx_right, 2: idx_up, 3: idx_down}[btn.cls_id]
                        sample_idx = int(np.random.choice(pool))
                        last_injected = CLASS_LABELS[btn.cls_id]
                        
                        pred_class, last_conf = run_inference(sample_idx)
                        last_prediction = CLASS_LABELS[pred_class]
                        last_status = "Matched" if pred_class == btn.cls_id else "Misclassified"
                        
                        if pred_class == 0: target_pos[0] -= STEP_SIZE
                        elif pred_class == 1: target_pos[0] += STEP_SIZE
                        elif pred_class == 2: target_pos[1] -= STEP_SIZE
                        elif pred_class == 3: target_pos[1] += STEP_SIZE
                        
                        target_pos[0] = max(360, min(960, target_pos[0]))
                        target_pos[1] = max(40, min(680, target_pos[1]))

    cursor_pos[0] += (target_pos[0] - cursor_pos[0]) * 0.2
    cursor_pos[1] += (target_pos[1] - cursor_pos[1]) * 0.2

    screen.fill((18, 20, 28))
    
    # UI Panels
    pygame.draw.rect(screen, (28, 32, 45), (20, 20, 300, 660), border_radius=10)
    screen.blit(font_title.render("4-Way Telemetry Deck", True, (255, 255, 255)), (40, 40))
    
    y_off = 100
    for label, val in [("Injected:", last_injected), ("Decoded:", last_prediction), 
                       ("Match:", last_status), ("Conf:", f"{last_conf:.1f}%" if last_conf else "--")]:
        screen.blit(font_body.render(label, True, (148, 163, 184)), (40, y_off))
        color = (34, 197, 94) if val == "Matched" else ((239, 68, 68) if val == "Misclassified" else (255, 255, 255))
        screen.blit(font_bold.render(str(val), True, color), (40, y_off + 20))
        y_off += 65

    for b in buttons: b.draw()

    # Grid Sandbox
    pygame.draw.rect(screen, (28, 32, 45), (340, 20, 640, 660), border_radius=10)
    pygame.draw.line(screen, (45, 55, 72), (660, 20), (660, 680)) # Vertical Axis
    pygame.draw.line(screen, (45, 55, 72), (340, 350), (980, 350)) # Horizontal Axis
    
    pygame.draw.circle(screen, (239, 68, 68), (int(cursor_pos[0]), int(cursor_pos[1])), 18)
    pygame.display.flip()
    clock.tick(60)