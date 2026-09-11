import cv2
import mediapipe as mp
import math
import time
import os
import queue
import threading
import collections
import numpy as np
try:
    import speech_recognition as sr
    HAS_SPEECH_REC = True
except ImportError:
    HAS_SPEECH_REC = False
USE_TASKS_API = not hasattr(mp, 'solutions')
if USE_TASKS_API:
    import urllib.request
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task') if '__file__' in globals() else 'hand_landmarker.task'
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model...")
        try:
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception as e:
            print(f"Warning downloading model: {e}")
    detector = None
    if os.path.exists(MODEL_PATH):
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error creating HandLandmarker: {e}")
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    def draw_landmarks(img, lms, color=(0, 255, 0)):
        ih, iw, _ = img.shape
        pts = [(int(lm.x * iw), int(lm.y * ih)) for lm in lms]
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(img, pts[p1], pts[p2], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
    def detect_hands(img, rgb_img):
        hands_list = []
        if detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            res = detector.detect(mp_image)
            if res.hand_landmarks:
                for i in range(len(res.hand_landmarks)):
                    lms = res.hand_landmarks[i]
                    label = 'Right'
                    if res.handedness and i < len(res.handedness) and res.handedness[i]:
                        label = res.handedness[i][0].category_name
                    draw_landmarks(img, lms, (0, 255, 0) if label == 'Right' else (255, 100, 0))
                    hands_list.append({'label': label, 'landmarks': lms})
        return hands_list
else:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    def detect_hands(img, rgb_img):
        res = hands.process(rgb_img)
        hands_list = []
        if res.multi_hand_landmarks:
            for i, hand_lms in enumerate(res.multi_hand_landmarks):
                label = 'Right'
                if res.multi_handedness and i < len(res.multi_handedness):
                    label = res.multi_handedness[i].classification[0].label
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                hands_list.append({'label': label, 'landmarks': hand_lms.landmark})
        return hands_list
class VoiceWorker(threading.Thread):
    def __init__(self, command_queue: queue.Queue):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.running = True
        self.status = "LISTENING"
    def run(self):
        if not HAS_SPEECH_REC:
            self.status = "DISABLED"
            return
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.6
        try:
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception as e:
            print(f"[VoiceWorker] Microphone init notice: {e}")
            self.status = "SIMULATED"
            return
        while self.running:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=3.0, phrase_time_limit=4.0)
                text = recognizer.recognize_google(audio).lower().strip()
                print(f"[VoiceWorker] Speech Recognized: '{text}'")
                self.command_queue.put(text)
            except Exception:
                time.sleep(0.5)
    def inject_command(self, cmd: str):
        self.command_queue.put(cmd.lower().strip())
    def stop(self):
        self.running = False
class MedicalTextureGenerator:
    @staticmethod
    def generate_mri_scan(width=780, height=560) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        cv2.ellipse(img, (cx, cy), (int(width * 0.35), int(height * 0.42)), 0, 0, 360, 180, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (int(width * 0.33), int(height * 0.40)), 0, 0, 360, 40, -1, cv2.LINE_AA)
        brain = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(brain, (cx, cy), (int(width * 0.32), int(height * 0.39)), 0, 0, 360, 120, -1, cv2.LINE_AA)
        np.random.seed(42)
        noise = np.random.normal(0, 25, (height, width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (15, 15), 0)
        brain_texture = cv2.add(brain.astype(np.float32), noise)
        brain_texture = np.clip(brain_texture, 0, 255).astype(np.uint8)
        brain_texture = cv2.bitwise_and(brain_texture, brain_texture, mask=brain)
        cv2.ellipse(brain_texture, (cx - 35, cy - 20), (20, 55), 15, 0, 360, 220, -1, cv2.LINE_AA)
        cv2.ellipse(brain_texture, (cx + 35, cy - 20), (20, 55), -15, 0, 360, 220, -1, cv2.LINE_AA)
        cv2.circle(brain_texture, (cx + 85, cy - 55), 32, 245, -1, cv2.LINE_AA)
        brain_texture = cv2.GaussianBlur(brain_texture, (5, 5), 0)
        cv2.circle(brain_texture, (cx + 85, cy - 55), 15, 255, -1, cv2.LINE_AA)
        final_mri = cv2.addWeighted(img, 0.3, brain_texture, 0.7, 0)
        color_mri = cv2.cvtColor(final_mri, cv2.COLOR_GRAY2BGR)
        color_mri[:, :, 0] = cv2.add(color_mri[:, :, 0], 25)
        return color_mri
    @staticmethod

    def generate_ct_scan(width=780, height=560) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        cv2.ellipse(img, (cx, cy), (int(width * 0.36), int(height * 0.42)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (int(width * 0.31), int(height * 0.37)), 0, 0, 360, 40, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (int(width * 0.30), int(height * 0.36)), 0, 0, 360, 90, -1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - int(height * 0.35)), (cx, cy + int(height * 0.35)), 160, 2, cv2.LINE_AA)
        cv2.circle(img, (cx - 40, cy + 150), 22, 10, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + 40, cy + 150), 22, 10, -1, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    @staticmethod

    def generate_xray_scan(width=780, height=560) -> np.ndarray:
        img = np.full((height, width), 30, dtype=np.uint8)
        cx, cy = width // 2, height // 2
        cv2.ellipse(img, (cx - 120, cy - 20), (95, 200), -10, 0, 360, 5, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 120, cy - 20), (95, 200), 10, 0, 360, 5, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 35, cy + 35), (100, 125), 30, 0, 360, 180, -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - 16, cy - 270), (cx + 16, cy + 270), 170, -1)
        for i in range(-4, 5):
            y_pos = cy + i * 38

            cv2.ellipse(img, (cx, y_pos), (230, 45), 0, 190, 350, 140, 5, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
HUD_W, HUD_H = 1280, 850
S = 2                           
HD_W, HD_H = HUD_W * S, HUD_H * S

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets') if '__file__' in globals() else 'assets'
CLINIC_BG_PATH = os.path.join(ASSETS_DIR, 'clinic.jpg')
X1_PATH = os.path.join(ASSETS_DIR, 'x1.jpg')
X2_PATH = os.path.join(ASSETS_DIR, 'x2.jpg')
clinic_bg_raw = cv2.imread(CLINIC_BG_PATH) if os.path.exists(CLINIC_BG_PATH) else None

if clinic_bg_raw is not None:
    bg_img_hd = cv2.resize(clinic_bg_raw, (HD_W, HD_H), interpolation=cv2.INTER_CUBIC)
    dark_tint = np.full_like(bg_img_hd, (15, 22, 34), dtype=np.uint8)
    bg_img_hd = cv2.addWeighted(bg_img_hd, 0.40, dark_tint, 0.60, 0)
    
else:
    bg_img_hd = np.full((HD_H, HD_W, 3), (15, 22, 34), dtype=np.uint8)
img_x1_raw = cv2.imread(X1_PATH) if os.path.exists(X1_PATH) else MedicalTextureGenerator.generate_mri_scan(780, 560)
img_x2_raw = cv2.imread(X2_PATH) if os.path.exists(X2_PATH) else MedicalTextureGenerator.generate_ct_scan(780, 560)
img_xray_raw = MedicalTextureGenerator.generate_xray_scan(780, 560)
thumb_x1 = cv2.resize(img_x1_raw, (300, 200), interpolation=cv2.INTER_CUBIC)
thumb_x2 = cv2.resize(img_x2_raw, (300, 200), interpolation=cv2.INTER_CUBIC)
STATE_MAIN = 0
STATE_FOLDER = 1
STATE_VIEW_IMAGE = 2
STATE_SLEEP = 3
current_state = STATE_FOLDER
selected_image_id = "x1"
last_voice_cmd = "System Ready. Listening..."
voice_cmd_log = collections.deque(maxlen=4)
voice_cmd_log.append("System Ready.")
voice_queue = queue.Queue()
voice_worker = VoiceWorker(voice_queue)
voice_worker.start()
choice_cards = [
    {"id": "folder",     "cx": 95, "cy": 140, "r": 40, "title": "ASSETS FOLDER", "subtitle": "Scans x1 & x2",  "color": (255, 180, 40)},
    {"id": "flip_cam",   "cx": 95, "cy": 250, "r": 40, "title": "FLIP CAMERA",   "subtitle": "Mirror Mode",   "color": (40, 180, 120)},
    {"id": "swap_hands", "cx": 95, "cy": 360, "r": 40, "title": "SWAP HANDS",    "subtitle": "Switch Control", "color": (160, 60, 220)},
]
action_buttons = [
    {"id": "zoom_mode", "cx": 560, "cy": 715, "r": 34, "label": "ZOOM MODE",  "color": (0, 180, 255)},
    {"id": "reset",     "cx": 720, "cy": 715, "r": 34, "label": "RESET VIEW", "color": (120, 140, 160)},
]
flip_camera = True
swap_hands = True
zoom_level = 1.0
pan_x, pan_y = 0, 0
brightness = 1.0
contrast = 1.0
zoom_mode_enabled = False
laser_enabled = False
surgical_pins = []
pin_counter = 1
last_clap_time = 0.0
sleep_clap_count = 0
sleep_last_clap_time = 0.0
last_click_time = 0.0
smooth_cursor_x, smooth_cursor_y = HUD_W // 2, HUD_H // 2
ALPHA = 0.70
MARGIN = 0.12
click_ripple_radius = 0
laser_trail = collections.deque(maxlen=18)
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)
def check_number_two_gesture(hand_lms):
    """Detects 'Number 2' Peace sign gesture (Index & Middle extended, Ring & Pinky folded)."""
    if not hand_lms:
        return False
    wrist = hand_lms[0]
    d_index  = dist(hand_lms[8], wrist)
    d_middle = dist(hand_lms[12], wrist)
    d_ring   = dist(hand_lms[16], wrist)
    d_pinky  = dist(hand_lms[20], wrist)
    return (d_index > 0.28 and d_middle > 0.28 and d_ring < 0.26 and d_pinky < 0.26)
def draw_icon_folder(img, cx, cy, color, S=2):
    pts = np.array([
        [cx - 22*S, cy - 11*S], [cx - 9*S, cy - 11*S], [cx - 2*S, cy - 18*S],
        [cx + 22*S, cy - 18*S], [cx + 22*S, cy + 18*S], [cx - 22*S, cy + 18*S]
    ], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (255, 255, 255), 2*S, cv2.LINE_AA)
    cv2.line(img, (cx - 12*S, cy - 2*S), (cx + 12*S, cy - 2*S), (20, 26, 38), 2*S, cv2.LINE_AA)
def draw_icon_flip(img, cx, cy, color, S=2):
    cv2.ellipse(img, (cx, cy), (20*S, 20*S), 0, 20, 160, color, 4*S, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (20*S, 20*S), 0, 200, 340, color, 4*S, cv2.LINE_AA)
def draw_icon_swap(img, cx, cy, color, S=2):
    cv2.arrowedLine(img, (cx - 20*S, cy - 9*S), (cx + 20*S, cy - 9*S), color, 4*S, tipLength=0.35)
    cv2.arrowedLine(img, (cx + 20*S, cy + 9*S), (cx - 20*S, cy + 9*S), color, 4*S, tipLength=0.35)
def draw_icon_zoom_mode(img, cx, cy, color, S=2):
    """Dual Zoom Mode Icon (Magnifying glass with + and - arrows)."""
    cv2.circle(img, (cx - 4*S, cy - 4*S), 13*S, color, 3*S, cv2.LINE_AA)
    cv2.line(img, (cx + 6*S, cy + 6*S), (cx + 18*S, cy + 18*S), color, 4*S, cv2.LINE_AA)
    cv2.line(img, (cx - 10*S, cy - 4*S), (cx + 2*S, cy - 4*S), (255, 255, 255), 2*S, cv2.LINE_AA)
    cv2.line(img, (cx - 4*S, cy - 10*S), (cx - 4*S, cy + 2*S), (255, 255, 255), 2*S, cv2.LINE_AA)
def draw_icon_laser(img, cx, cy, color, S=2):
    cv2.circle(img, (cx, cy), 12*S, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 18*S, (255, 255, 255), 2*S, cv2.LINE_AA)
def draw_icon_reset(img, cx, cy, color, S=2):
    cv2.ellipse(img, (cx, cy), (16*S, 16*S), 0, 30, 300, color, 3*S, cv2.LINE_AA)
    pts = np.array([[cx + 10*S, cy - 16*S], [cx + 20*S, cy - 6*S], [cx + 18*S, cy - 20*S]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
def draw_button_icon(hud, icon_id, cx, cy, color, S=2):
    if icon_id == "folder":
        draw_icon_folder(hud, cx, cy, color, S)
    elif icon_id == "flip_cam":
        draw_icon_flip(hud, cx, cy, color, S)
    elif icon_id == "swap_hands":
        draw_icon_swap(hud, cx, cy, color, S)
    elif icon_id == "zoom_mode":
        draw_icon_zoom_mode(hud, cx, cy, color, S)
    elif icon_id == "laser":
        draw_icon_laser(hud, cx, cy, color, S)
    elif icon_id == "reset":
        draw_icon_reset(hud, cx, cy, color, S)
    else:
        cv2.circle(hud, (cx, cy), 14*S, color, 3*S, cv2.LINE_AA)
cap = cv2.VideoCapture(0)
use_synthetic_cam = False
if not cap.isOpened():
    print("[Main] Physical camera absent. Enabling synthetic camera generator.")
    use_synthetic_cam = True
cv2.namedWindow("OR-Nav Camera Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Interactive Virtual Control Screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Interactive Virtual Control Screen", HUD_W, HUD_H)
frame_start_time = time.time()
fps_counter = 60.0
synthetic_angle = 0.0
while True:
    t_start = time.time()
    try:
        while not voice_queue.empty():
            cmd = voice_queue.get_nowait()
            last_voice_cmd = cmd
            voice_cmd_log.append(f"VOICE: '{cmd}'")
            print(f"[Main] Executing Voice Command: '{cmd}'")
            if "open mri" in cmd or "mri scan" in cmd or "open x1" in cmd:
                selected_image_id = "x1"
                current_state = STATE_VIEW_IMAGE
                voice_cmd_log.append("VOICE: Opened MRI Scan.")
            elif "open ct" in cmd or "ct scan" in cmd or "cranial" in cmd or "open x2" in cmd:
                selected_image_id = "x2"
                current_state = STATE_VIEW_IMAGE
                voice_cmd_log.append("VOICE: Opened CT Scan.")
            elif "open scan" in cmd or "open xray" in cmd or "xray" in cmd:
                selected_image_id = "x1"
                current_state = STATE_VIEW_IMAGE
                voice_cmd_log.append("VOICE: Opened X-Ray Scan.")
            elif "open folder" in cmd or "folder" in cmd or "assets" in cmd or "scans" in cmd:
                current_state = STATE_FOLDER
                voice_cmd_log.append("VOICE: Opened Assets Folder.")
            elif "close" in cmd or "back" in cmd or "exit" in cmd or "return" in cmd:
                if current_state == STATE_VIEW_IMAGE:
                    current_state = STATE_FOLDER
                    zoom_mode_enabled = False
                    voice_cmd_log.append("VOICE: Closed Scan -> Folder.")
                elif current_state == STATE_FOLDER:
                    current_state = STATE_MAIN
                    voice_cmd_log.append("VOICE: Closed Folder -> Main.")
            elif "zoom in" in cmd or cmd == "in" or "enlarge" in cmd or "bigger" in cmd:
                zoom_level = min(3.5, zoom_level + 0.4)
                voice_cmd_log.append(f"VOICE: Zoomed In ({zoom_level:.1f}x)")
            elif "zoom out" in cmd or cmd == "out" or "smaller" in cmd or "shrink" in cmd:
                zoom_level = max(1.0, zoom_level - 0.4)
                voice_cmd_log.append(f"VOICE: Zoomed Out ({zoom_level:.1f}x)")
            elif "zoom mode" in cmd or "toggle zoom" in cmd or "enable zoom" in cmd or "disable zoom" in cmd:
                zoom_mode_enabled = not zoom_mode_enabled
                voice_cmd_log.append(f"VOICE: Zoom Mode {'ENABLED' if zoom_mode_enabled else 'DISABLED'}")
            elif "zoom here" in cmd or "magnify" in cmd or "focus here" in cmd:
                pan_x = int((640 - smooth_cursor_x))
                pan_y = int((380 - smooth_cursor_y))
                zoom_level = min(3.5, zoom_level * 1.5)
                voice_cmd_log.append(f"POINT&SPEAK: Zoomed target ({smooth_cursor_x},{smooth_cursor_y})")
            elif "flip camera" in cmd or "flip" in cmd or "mirror" in cmd:
                flip_camera = not flip_camera
                voice_cmd_log.append(f"VOICE: Camera Flip {'ON' if flip_camera else 'OFF'}")
            elif "swap hands" in cmd or "swap hand" in cmd or "switch hand" in cmd:
                swap_hands = not swap_hands
                voice_cmd_log.append(f"VOICE: Swap Hands {'ON' if swap_hands else 'OFF'}")
            elif "toggle laser" in cmd or "laser" in cmd or "pointer" in cmd:
                laser_enabled = not laser_enabled
                voice_cmd_log.append(f"VOICE: Laser {'ENABLED' if laser_enabled else 'DISABLED'}")
            elif "pin this" in cmd or "mark" in cmd or "pin" in cmd or "add pin" in cmd:
                surgical_pins.append({"id": pin_counter, "x": smooth_cursor_x, "y": smooth_cursor_y})
                voice_cmd_log.append(f"VOICE: PIN #{pin_counter} at ({smooth_cursor_x},{smooth_cursor_y})")
                pin_counter += 1
            elif "clear pins" in cmd or "remove pins" in cmd or "delete pins" in cmd:
                surgical_pins.clear()
                voice_cmd_log.append("VOICE: All Pins Cleared.")
            elif "pan left" in cmd or "move left" in cmd:
                pan_x += 60
                voice_cmd_log.append("VOICE: Panned Left.")
            elif "pan right" in cmd or "move right" in cmd:
                pan_x -= 60
                voice_cmd_log.append("VOICE: Panned Right.")
            elif "pan up" in cmd or "move up" in cmd:
                pan_y += 60
                voice_cmd_log.append("VOICE: Panned Up.")
            elif "pan down" in cmd or "move down" in cmd:
                pan_y -= 60
                voice_cmd_log.append("VOICE: Panned Down.")
            elif "brighter" in cmd or "brightness up" in cmd:
                brightness = min(2.0, brightness + 0.2)
                voice_cmd_log.append(f"VOICE: Brightness {brightness:.1f}")
            elif "darker" in cmd or "brightness down" in cmd:
                brightness = max(0.4, brightness - 0.2)
                voice_cmd_log.append(f"VOICE: Brightness {brightness:.1f}")
            elif "reset" in cmd or "reset view" in cmd or "home" in cmd:
                zoom_level, pan_x, pan_y, brightness, contrast = 1.0, 0, 0, 1.0, 1.0
                zoom_mode_enabled = False
                voice_cmd_log.append("VOICE: View Reset.")
            elif "sleep" in cmd or "standby" in cmd:
                current_state = STATE_SLEEP
                voice_cmd_log.append("VOICE: Entered Sleep Mode.")
            elif "wake up" in cmd or "wake" in cmd or "activate" in cmd:
                current_state = STATE_MAIN
                voice_cmd_log.append("VOICE: System Woken Up.")
    except Exception:
        pass
    if not use_synthetic_cam and cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            use_synthetic_cam = True
            continue
        if flip_camera:
            frame = cv2.flip(frame, 1)
    else:
        synthetic_angle += 0.04
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (630, 470), (30, 41, 59), 2, cv2.LINE_AA)
        cv2.putText(frame, "SIMULATED CAMERA STREAM", (160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 148, 14), 2, cv2.LINE_AA)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_list = detect_hands(frame, rgb_frame)
    cursor_hand_lms = None
    cursor_label = "NO HAND DETECTED"
    if hands_list:
        hand_info = hands_list[0]
        cursor_hand_lms = hand_info['landmarks']
        cursor_label = f"{hand_info['label'].upper()} HAND"
        if len(hands_list) > 1 and swap_hands:
            cursor_hand_lms = hands_list[1]['landmarks']
            cursor_label = f"{hands_list[1]['label'].upper()} HAND"
    is_clicking = False
    is_laser_gesture = False
    if cursor_hand_lms:
        thumb_tip = cursor_hand_lms[4]
        index_tip = cursor_hand_lms[8]
        middle_tip = cursor_hand_lms[12]
        c_wrist   = cursor_hand_lms[0]
        pinch_click = dist(thumb_tip, index_tip) < 0.06
        c_index_folded  = dist(cursor_hand_lms[8], c_wrist) < 0.25
        c_middle_folded = dist(cursor_hand_lms[12], c_wrist) < 0.25
        c_ring_folded   = dist(cursor_hand_lms[16], c_wrist) < 0.25
        c_pinky_folded  = dist(cursor_hand_lms[20], c_wrist) < 0.25
        fist_click = c_index_folded and c_middle_folded and c_ring_folded and c_pinky_folded
        is_clicking = pinch_click or fist_click
        if dist(index_tip, c_wrist) > 0.28 and dist(middle_tip, c_wrist) < 0.25:
            is_laser_gesture = True
    if not is_clicking and len(hands_list) > 1:
        other_hand_lms = hands_list[1]['landmarks'] if cursor_hand_lms == hands_list[0]['landmarks'] else hands_list[0]['landmarks']
        o_pinch = dist(other_hand_lms[4], other_hand_lms[8]) < 0.06
        if o_pinch:
            is_clicking = True
    if cursor_hand_lms:
        c_index_tip = cursor_hand_lms[8]
        norm_x = np.clip((c_index_tip.x - MARGIN) / (1.0 - 2 * MARGIN), 0.0, 1.0)
        norm_y = np.clip((c_index_tip.y - MARGIN) / (1.0 - 2 * MARGIN), 0.0, 1.0)
        raw_cx = int(norm_x * HUD_W)
        raw_cy = int(norm_y * HUD_H)
        smooth_cursor_x = int(ALPHA * raw_cx + (1 - ALPHA) * smooth_cursor_x)
        smooth_cursor_y = int(ALPHA * raw_cy + (1 - ALPHA) * smooth_cursor_y)
        smooth_cursor_x = max(0, min(HUD_W - 1, smooth_cursor_x))
        smooth_cursor_y = max(0, min(HUD_H - 1, smooth_cursor_y))
    elif use_synthetic_cam:
        synth_norm_x = (320 + 200 * math.sin(synthetic_angle)) / 640.0
        synth_norm_y = (240 + 100 * math.sin(2 * synthetic_angle)) / 480.0
        smooth_cursor_x = int(synth_norm_x * HUD_W)
        smooth_cursor_y = int(synth_norm_y * HUD_H)
        cursor_label = "SYNTHETIC HAND"
    laser_trail.append((smooth_cursor_x, smooth_cursor_y))
    is_gesture_two = any(check_number_two_gesture(h['landmarks']) for h in hands_list)
    curr_time = time.time()
    if is_gesture_two and (curr_time - last_clap_time > 0.6):
        last_clap_time = curr_time
        click_ripple_radius = 50
        if current_state == STATE_VIEW_IMAGE:
            current_state = STATE_FOLDER
            zoom_mode_enabled = False
            voice_cmd_log.append("GESTURE: Closed scan -> Folder.")
        elif current_state == STATE_FOLDER:
            current_state = STATE_MAIN
            voice_cmd_log.append("GESTURE: Closed folder -> Main.")
        elif current_state == STATE_MAIN:
            current_state = STATE_SLEEP
            sleep_clap_count = 0
            sleep_last_clap_time = 0.0
            voice_cmd_log.append("GESTURE: Entered Sleep Mode.")
        elif current_state == STATE_SLEEP:
            if curr_time - sleep_last_clap_time < 1.8:
                sleep_clap_count += 1
            else:
                sleep_clap_count = 1
            sleep_last_clap_time = curr_time
            if sleep_clap_count >= 2:
                current_state = STATE_MAIN
                sleep_clap_count = 0
                voice_cmd_log.append("GESTURE: System Woken Up.")
    if current_state == STATE_SLEEP:
        hud_hd = np.full((HD_H, HD_W, 3), (8, 10, 16), dtype=np.uint8)
        cv2.circle(hud_hd, ((HUD_W // 2)*S, (HUD_H // 2 - 45)*S), 75*S, (255, 180, 40), 4*S, cv2.LINE_AA)
        cv2.circle(hud_hd, ((HUD_W // 2)*S, (HUD_H // 2 - 45)*S), 25*S, (255, 180, 40), -1, cv2.LINE_AA)
        cv2.putText(hud_hd, "SYSTEM STANDBY / STERILE SLEEP MODE", ((HUD_W // 2 - 320)*S, (HUD_H // 2 + 55)*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95*S, (255, 255, 255), 2*S, cv2.LINE_AA)
        gesture_text = "✌️ 1/2 GESTURE DETECTED! SHOW ✌️ AGAIN TO WAKE UP!" if sleep_clap_count == 1 else "SHOW ✌️ GESTURE (NO. 2) TWICE TO WAKE UP SYSTEM"
        cv2.putText(hud_hd, gesture_text, ((HUD_W // 2 - 350)*S, (HUD_H // 2 + 120)*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65*S, (0, 255, 200) if sleep_clap_count == 1 else (180, 195, 210), 2*S, cv2.LINE_AA)
        hud = cv2.resize(hud_hd, (HUD_W, HUD_H), interpolation=cv2.INTER_AREA)
        cv2.imshow("OR-Nav Camera Feed", frame)
        cv2.imshow("Interactive Virtual Control Screen", hud)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    hud_hd = bg_img_hd.copy()
    sub_hdr = hud_hd[0:80*S, 0:HD_W]
    hdr_overlay = np.full_like(sub_hdr, (25, 34, 48), dtype=np.uint8)
    hud_hd[0:80*S, 0:HD_W] = cv2.addWeighted(sub_hdr, 0.20, hdr_overlay, 0.80, 0)
    cv2.putText(hud_hd, "❖ GESTIMED VOXMOTION | SURGICAL INTERFACE v2.4", (22*S, 34*S),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70*S, (0, 255, 200), 2*S, cv2.LINE_AA)
    patient_text = "PATIENT: John Doe (M, 54)  |  BP: 120/80  •  HR: 72 bpm  |  PROCEDURE: Parietal Craniotomy"
    cv2.putText(hud_hd, patient_text, (22*S, 62*S),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44*S, (220, 230, 240), 1*S, cv2.LINE_AA)
    t_end = time.time()
    latency_val = (t_end - t_start) * 1000.0
    dt_frame = t_end - frame_start_time
    frame_start_time = t_end
    if dt_frame > 0:
        fps_counter = 0.9 * fps_counter + 0.1 * (1.0 / dt_frame)
    cv2.putText(hud_hd, f"CAM: LIVE 🟢  MIC: {voice_worker.status} 🟢  STERILE ACTIVE  FPS: {fps_counter:.1f}  LATENCY: {latency_val:.1f}ms",
                (640*S, 34*S), cv2.FONT_HERSHEY_SIMPLEX, 0.44*S, (0, 255, 200), 1*S, cv2.LINE_AA)
    cv2.putText(hud_hd, f"LAST VOICE: '{last_voice_cmd}'",
                (640*S, 62*S), cv2.FONT_HERSHEY_SIMPLEX, 0.44*S, (255, 180, 0), 1*S, cv2.LINE_AA)
    for card in choice_cards:
        cx, cy, r = card["cx"], card["cy"], card["r"]
        dist_to_cursor = math.hypot(smooth_cursor_x - cx, smooth_cursor_y - cy)
        is_hovered = (dist_to_cursor <= r + 4)
        is_selected = (card["id"] == "folder" and current_state in [STATE_FOLDER, STATE_VIEW_IMAGE])
        cur_r = r + 5 if (is_hovered or is_selected) else r
        circle_bg = (65, 95, 140) if is_selected else ((48, 68, 98) if is_hovered else (28, 36, 50))
        cv2.circle(hud_hd, (cx*S, cy*S), cur_r*S, circle_bg, -1, cv2.LINE_AA)
        border_color = (0, 255, 200) if is_selected else ((255, 255, 255) if is_hovered else (70, 90, 120))
        cv2.circle(hud_hd, (cx*S, cy*S), cur_r*S, border_color, (3 if (is_selected or is_hovered) else 2)*S, cv2.LINE_AA)
        draw_button_icon(hud_hd, card["id"], cx*S, cy*S, (0, 255, 200) if is_selected else (255, 255, 255), S)
        tx, ty = cx + cur_r + 14, cy + 5
        tw = len(card["title"]) * 11
        cv2.rectangle(hud_hd, ((tx - 6)*S, (ty - 16)*S), ((tx + tw)*S, (ty + 8)*S), (18, 24, 36), -1)
        cv2.rectangle(hud_hd, ((tx - 6)*S, (ty - 16)*S), ((tx + tw)*S, (ty + 8)*S), border_color if (is_hovered or is_selected) else (50, 65, 85), 1*S, cv2.LINE_AA)
        cv2.putText(hud_hd, card["title"], (tx*S, ty*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46*S, (255, 255, 255) if (is_hovered or is_selected) else (180, 195, 210), 1*S, cv2.LINE_AA)
        if is_hovered and is_clicking and (curr_time - last_click_time > 0.35):
            last_click_time = curr_time
            click_ripple_radius = 25
            if card["id"] == "flip_cam":
                flip_camera = not flip_camera
            elif card["id"] == "swap_hands":
                swap_hands = not swap_hands
            elif card["id"] == "folder":
                current_state = STATE_FOLDER
    center_x, center_y = 640, 380
    if current_state == STATE_FOLDER:
        sub_fold = hud_hd[95*S:745*S, 230*S:1220*S]
        fold_bg = np.full_like(sub_fold, (18, 24, 36), dtype=np.uint8)
        hud_hd[95*S:745*S, 230*S:1220*S] = cv2.addWeighted(sub_fold, 0.2, fold_bg, 0.8, 0)
        cv2.rectangle(hud_hd, (230*S, 95*S), (1220*S, 745*S), (255, 180, 40), 2*S, cv2.LINE_AA)
        cv2.putText(hud_hd, "📁 ASSETS MEDICAL SCANS FOLDER", (260*S, 140*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80*S, (255, 180, 40), 2*S, cv2.LINE_AA)
        cv2.putText(hud_hd, "SHOW ✌️ GESTURE (NO. 2) TO CLOSE FOLDER | CLICK SCAN TO OPEN:", (260*S, 175*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45*S, (180, 195, 210), 1*S, cv2.LINE_AA)
        cx1, cy1 = 450, 420
        is_hov_1 = (cx1 - 150 <= smooth_cursor_x <= cx1 + 150 and cy1 - 100 <= smooth_cursor_y <= cy1 + 100)
        border_1 = (0, 255, 200) if is_hov_1 else (70, 90, 120)
        cv2.rectangle(hud_hd, ((cx1 - 152)*S, (cy1 - 105)*S), ((cx1 + 152)*S, (cy1 + 145)*S), (24, 32, 48), -1)
        cv2.rectangle(hud_hd, ((cx1 - 152)*S, (cy1 - 105)*S), ((cx1 + 152)*S, (cy1 + 145)*S), border_1, (3 if is_hov_1 else 1)*S, cv2.LINE_AA)
        thumb1_hd = cv2.resize(thumb_x1, (300*S, 200*S), interpolation=cv2.INTER_CUBIC)
        hud_hd[(cy1 - 100)*S:(cy1 + 100)*S, (cx1 - 150)*S:(cx1 + 150)*S] = thumb1_hd
        cv2.putText(hud_hd, "MRI BRAIN SCAN (X1)", ((cx1 - 95)*S, (cy1 + 130)*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52*S, (255, 255, 255), 2*S, cv2.LINE_AA)
        if is_hov_1 and is_clicking and (curr_time - last_click_time > 0.35):
            last_click_time = curr_time
            current_state = STATE_VIEW_IMAGE
            selected_image_id = "x1"
        cx2, cy2 = 900, 420
        is_hov_2 = (cx2 - 150 <= smooth_cursor_x <= cx2 + 150 and cy2 - 100 <= smooth_cursor_y <= cy2 + 100)
        border_2 = (0, 255, 200) if is_hov_2 else (70, 90, 120)
        cv2.rectangle(hud_hd, ((cx2 - 152)*S, (cy2 - 105)*S), ((cx2 + 152)*S, (cy2 + 145)*S), (24, 32, 48), -1)
        cv2.rectangle(hud_hd, ((cx2 - 152)*S, (cy2 - 105)*S), ((cx2 + 152)*S, (cy2 + 145)*S), border_2, (3 if is_hov_2 else 1)*S, cv2.LINE_AA)
        thumb2_hd = cv2.resize(thumb_x2, (300*S, 200*S), interpolation=cv2.INTER_CUBIC)
        hud_hd[(cy2 - 100)*S:(cy2 + 100)*S, (cx2 - 150)*S:(cx2 + 150)*S] = thumb2_hd
        cv2.putText(hud_hd, "CT CRANIAL SCAN (X2)", ((cx2 - 95)*S, (cy2 + 130)*S),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52*S, (255, 255, 255), 2*S, cv2.LINE_AA)
        if is_hov_2 and is_clicking and (curr_time - last_click_time > 0.35):
            last_click_time = curr_time
            current_state = STATE_VIEW_IMAGE
            selected_image_id = "x2"
    elif current_state == STATE_VIEW_IMAGE:
        zoom_status_str = f"ZOOM: {zoom_level:.1f}x (INACTIVE - CLICK 'ZOOM MODE' TO ENABLE)"
        if zoom_mode_enabled:
            zoom_status_str = f"🔍 ZOOM MODE ACTIVE ({zoom_level:.1f}x)"
            if cursor_hand_lms:
                c_wrist  = cursor_hand_lms[0]
                d_thumb  = dist(cursor_hand_lms[4], c_wrist)
                d_index  = dist(cursor_hand_lms[8], c_wrist)
                d_middle = dist(cursor_hand_lms[12], c_wrist)
                d_ring   = dist(cursor_hand_lms[16], c_wrist)
                d_pinky  = dist(cursor_hand_lms[20], c_wrist)
                avg_dist = (d_thumb + d_index + d_middle + d_ring + d_pinky) / 5.0
                hand_span = dist(cursor_hand_lms[4], cursor_hand_lms[20])
                if hand_span > 0.28 and avg_dist > 0.28:
                    zoom_level = min(3.5, zoom_level + 0.04)
                    zoom_status_str = f"✋ OPEN HAND: ZOOMING IN ({zoom_level:.1f}x)"
                elif hand_span < 0.20 or avg_dist < 0.22:
                    zoom_level = max(0.5, zoom_level - 0.04)
                    zoom_status_str = f"✊ CLOSED HAND: ZOOMING OUT ({zoom_level:.1f}x)"
        target_img = img_x1_raw if selected_image_id == "x1" else (img_x2_raw if selected_image_id == "x2" else img_xray_raw)
        adjusted = cv2.convertScaleAbs(target_img, alpha=contrast, beta=float((brightness - 1.0) * 128.0))
        vw, vh = int(780 * zoom_level), int(560 * zoom_level)
        sw_hd, sh_hd = vw * S, vh * S
        viewport_canvas = np.zeros((560 * S, 780 * S, 3), dtype=np.uint8)
        resized_hd = cv2.resize(adjusted, (sw_hd, sh_hd), interpolation=cv2.INTER_CUBIC)
        V_W, V_H = 780 * S, 560 * S
        scx, scy = (V_W // 2) + pan_x * S, (V_H // 2) + pan_y * S
        img_x1 = scx - (sw_hd // 2)
        img_y1 = scy - (sh_hd // 2)
        img_x2 = img_x1 + sw_hd
        img_y2 = img_y1 + sh_hd
        c_x1 = max(0, img_x1)
        c_y1 = max(0, img_y1)
        c_x2 = min(V_W, img_x2)
        c_y2 = min(V_H, img_y2)
        if c_x2 > c_x1 and c_y2 > c_y1:
            s_x1 = c_x1 - img_x1
            s_y1 = c_y1 - img_y1
            s_x2 = s_x1 + (c_x2 - c_x1)
            s_y2 = s_y1 + (c_y2 - c_y1)
            viewport_canvas[c_y1:c_y2, c_x1:c_x2] = resized_hd[s_y1:s_y2, s_x1:s_x2]
        hud_hd[95*S:655*S, 250*S:1030*S] = viewport_canvas
        cv2.rectangle(hud_hd, (250*S, 95*S), (1030*S, 655*S), (0, 255, 200), 2*S, cv2.LINE_AA)
        for pin in surgical_pins:
            px, py = pin['x']*S, pin['y']*S
            if 250*S <= px < 1030*S and 95*S <= py < 655*S:
                cv2.circle(hud_hd, (px, py), 14*S, (0, 255, 200), 2*S, cv2.LINE_AA)
                cv2.line(hud_hd, (px - 20*S, py), (px + 20*S, py), (0, 255, 200), 1*S, cv2.LINE_AA)
                cv2.line(hud_hd, (px, py - 20*S), (px, py + 20*S), (0, 255, 200), 1*S, cv2.LINE_AA)
                cv2.putText(hud_hd, f"PIN #{pin['id']}", (px + 16*S, py - 8*S),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4*S, (255, 180, 0), 1*S, cv2.LINE_AA)
        img_name = "MRI_BRAIN" if selected_image_id == "x1" else ("CT_CRANIAL" if selected_image_id == "x2" else "XRAY_CHEST")
        cv2.putText(hud_hd, f"🔍 SCAN: {img_name} | {zoom_status_str} | ✌️ TO CLOSE",
                    (260*S, 125*S), cv2.FONT_HERSHEY_SIMPLEX, 0.45*S, (0, 255, 200) if zoom_mode_enabled else (180, 195, 210), 2*S, cv2.LINE_AA)
        for btn in action_buttons:
            cx, cy, r = btn["cx"], btn["cy"], btn["r"]
            dist_to_cursor = math.hypot(smooth_cursor_x - cx, smooth_cursor_y - cy)
            is_hovered = (dist_to_cursor <= r + 4)
            is_active_toggle = (btn["id"] == "zoom_mode" and zoom_mode_enabled)
            cur_r = r + 4 if (is_hovered or is_active_toggle) else r
            btn_color = (0, 255, 200) if (btn["id"] == "zoom_mode" and zoom_mode_enabled) else btn["color"]
            if is_hovered:
                btn_color = tuple(min(255, c + 50) for c in btn_color)
            cv2.circle(hud_hd, (cx*S, cy*S), cur_r*S, btn_color, -1, cv2.LINE_AA)
            border_color = (0, 255, 200) if is_active_toggle else ((255, 255, 255) if is_hovered else (90, 110, 140))
            cv2.circle(hud_hd, (cx*S, cy*S), cur_r*S, border_color, (3 if (is_hovered or is_active_toggle) else 2)*S, cv2.LINE_AA)
            draw_button_icon(hud_hd, btn["id"], cx*S, cy*S, (0, 255, 200) if is_active_toggle else (255, 255, 255), S)
            tx = cx - 42
            ty = cy + cur_r + 18
            tw = len(btn["label"]) * 8
            cv2.rectangle(hud_hd, ((tx - 4)*S, (ty - 14)*S), ((tx + tw + 8)*S, (ty + 6)*S), (18, 24, 36), -1)
            cv2.rectangle(hud_hd, ((tx - 4)*S, (ty - 14)*S), ((tx + tw + 8)*S, (ty + 6)*S), border_color if (is_hovered or is_active_toggle) else (50, 65, 85), 1*S, cv2.LINE_AA)
            cv2.putText(hud_hd, btn["label"], (tx*S, ty*S),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38*S, (0, 255, 200) if is_active_toggle else ((255, 255, 255) if is_hovered else (200, 210, 220)), 1*S, cv2.LINE_AA)
            if is_hovered and is_clicking and (curr_time - last_click_time > 0.35):
                last_click_time = curr_time
                click_ripple_radius = 25
                if btn["id"] == "zoom_mode":
                    zoom_mode_enabled = not zoom_mode_enabled
                    voice_cmd_log.append(f"ZOOM MODE: {'ENABLED' if zoom_mode_enabled else 'DISABLED'}")
                elif btn["id"] == "reset":
                    zoom_level, pan_x, pan_y, brightness, contrast = 1.0, 0, 0, 1.0, 1.0
                    zoom_mode_enabled = False
    if is_laser_gesture and len(laser_trail) > 1:
        for i in range(1, len(laser_trail)):
            cv2.line(hud_hd, (laser_trail[i - 1][0]*S, laser_trail[i - 1][1]*S),
                     (laser_trail[i][0]*S, laser_trail[i][1]*S), (0, 0, 255), 3*S, cv2.LINE_AA)
        cv2.circle(hud_hd, (smooth_cursor_x*S, smooth_cursor_y*S), 14*S, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(hud_hd, (smooth_cursor_x*S, smooth_cursor_y*S), 6*S, (255, 255, 255), -1, cv2.LINE_AA)
    sub_telem = hud_hd[780*S:845*S, 30*S:1250*S]
    telem_overlay = np.full_like(sub_telem, (28, 36, 50), dtype=np.uint8)
    hud_hd[780*S:845*S, 30*S:1250*S] = cv2.addWeighted(sub_telem, 0.25, telem_overlay, 0.75, 0)
    zoom_hint = "ZOOM MODE: ON (✋=IN ✊=OUT)" if zoom_mode_enabled else "ZOOM MODE: OFF (CLICK 'ZOOM MODE' BUTTON TO TOGGLE)"
    cv2.putText(hud_hd, f"CONTROL MODE: {cursor_label} = POINTER & CLICK | {zoom_hint} | ✌️ PEACE = CLOSE/SLEEP",
                (45*S, 804*S), cv2.FONT_HERSHEY_SIMPLEX, 0.43*S, (255, 255, 255), 1*S, cv2.LINE_AA)
    cv2.putText(hud_hd, f"CLICK GESTURE: {'ACTIVE' if is_clicking else 'INACTIVE'} | CURSOR: ({smooth_cursor_x},{smooth_cursor_y}) | VOICE LOG: {list(voice_cmd_log)[-1]}",
                (45*S, 830*S), cv2.FONT_HERSHEY_SIMPLEX, 0.43*S, (0, 255, 0) if is_clicking else (170, 180, 190), 1*S, cv2.LINE_AA)
    if click_ripple_radius > 0:
        cv2.circle(hud_hd, (smooth_cursor_x*S, smooth_cursor_y*S), click_ripple_radius*S, (0, 255, 0), 2*S, cv2.LINE_AA)
        click_ripple_radius += 3
        if click_ripple_radius > 45:
            click_ripple_radius = 0
    cursor_color = (0, 255, 0) if is_clicking else (0, 220, 255)
    cv2.circle(hud_hd, (smooth_cursor_x*S, smooth_cursor_y*S), (8 if is_clicking else 6)*S, cursor_color, -1, cv2.LINE_AA)
    cv2.circle(hud_hd, (smooth_cursor_x*S, smooth_cursor_y*S), 16*S, cursor_color, 2*S, cv2.LINE_AA)
    cv2.line(hud_hd, ((smooth_cursor_x - 22)*S, smooth_cursor_y*S), ((smooth_cursor_x + 22)*S, smooth_cursor_y*S), cursor_color, 1*S, cv2.LINE_AA)
    cv2.line(hud_hd, (smooth_cursor_x*S, (smooth_cursor_y - 22)*S), (smooth_cursor_x*S, (smooth_cursor_y + 22)*S), cursor_color, 1*S, cv2.LINE_AA)
    hud = cv2.resize(hud_hd, (HUD_W, HUD_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (10, 10), (620, 75), (20, 20, 20), -1)
    cv2.putText(frame, f"CURSOR: {cursor_label} ({smooth_cursor_x},{smooth_cursor_y})", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"CLICK: {'ACTIVE (PINCH/FIST)' if is_clicking else 'OPEN'} | VOICE: {last_voice_cmd}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if is_clicking else (180, 180, 180), 2, cv2.LINE_AA)
    cv2.imshow("OR-Nav Camera Feed", frame)
    cv2.imshow("Interactive Virtual Control Screen", hud)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
if cap and cap.isOpened():
    cap.release()
voice_worker.stop()
cv2.destroyAllWindows()
