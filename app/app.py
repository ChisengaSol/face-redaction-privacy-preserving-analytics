import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO

import forbes
from database import AnalyticsDB
from face_align import align_face

# --- Shared frame buffer for MJPEG streaming ---
output_frame = None
frame_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if output_frame is None:
                            time.sleep(0.033)
                            continue
                        _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_bytes = buffer.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)
            except Exception:
                pass

    def log_message(self, format, *args):
        pass  # suppress per-request logs


def start_mjpeg_server():
    server = HTTPServer(('0.0.0.0', 8001), MJPEGHandler)
    server.serve_forever()

# 1. Setup Models & Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assumes you are running this from inside the 'app' folder
detector = YOLO("../yolo_detector/weights/best.pt")

# Forbes Config
ADAFACE_CKPT = "adaface_ir101_ms1mv2.ckpt"
optimizer_brs = forbes.init_forbes(ADAFACE_CKPT, device)

# UAP Config — load pre-computed delta if available
UAP_PATH = "uap_delta.npy"
# Number of mosaic blocks per side on the 112×112 working image.
# 5 blocks → ~22×22px tiles; faces become completely unrecognisable.
# MUST match the value in compute_uap.py.
UAP_BLOCKS = 5
uap_delta = None
try:
    _arr = np.load(UAP_PATH)
    uap_delta = torch.from_numpy(_arr).float().to(device)
    print(f"UAP delta loaded from '{UAP_PATH}' — using real-time mode.")
except FileNotFoundError:
    print(f"No UAP delta found at '{UAP_PATH}' — falling back to Forbes (slow).")
    print("Run compute_uap.py once to enable real-time obfuscation.")

class FaceGallery:
    def __init__(self, encoder_model, threshold=0.45):
        self.database = {}  # {Global_ID: embedding_tensor}
        self.next_id = 1
        self.threshold = threshold
        self.encoder_model = encoder_model

    def load_from_db(self, db_instance):
        """Restore all previously identified people from the database on startup."""
        records = db_instance.load_embeddings()
        for person_id, embedding_list in records:
            tensor = torch.tensor(embedding_list, dtype=torch.float32).unsqueeze(0).to(device)
            self.database[person_id] = tensor
        if self.database:
            self.next_id = max(self.database.keys()) + 1
        print(f"Loaded {len(self.database)} known identities from previous sessions.")

    def match_or_create(self, face_tensor, db_instance=None):
        with torch.no_grad():
            embedding, _ = self.encoder_model(face_tensor.to(device))

        emb_norm = embedding.norm().item()

        all_sims = {pid: torch.nn.functional.cosine_similarity(embedding, e).item()
                    for pid, e in self.database.items()}

        best_id  = max(all_sims, key=all_sims.get) if all_sims else None
        max_similarity = all_sims[best_id] if best_id is not None else -1.0

        sim_str = "  ".join(f"P{pid}={s:.3f}" for pid, s in sorted(all_sims.items()))
        print(f"[ReID] norm={emb_norm:.3f}  sims=[{sim_str}]  best={max_similarity:.3f}")

        if max_similarity > self.threshold:
            updated = 0.85 * self.database[best_id] + 0.15 * embedding
            self.database[best_id] = torch.nn.functional.normalize(updated, dim=1)
            if db_instance:
                emb_list = self.database[best_id].cpu().numpy().flatten().tolist()
                db_instance.log_embedding(best_id, emb_list)
            print(f"[ReID] → Matched Person {best_id}")
            return best_id

        new_id = self.next_id
        self.database[new_id] = embedding
        self.next_id += 1
        print(f"[ReID] → New Person {new_id}")

        if db_instance:
            emb_list = embedding.cpu().numpy().flatten().tolist()
            db_instance.log_embedding(new_id, emb_list)

        return new_id

def preprocess_for_adaface(face_img):
    aligned = align_face(face_img)   # MediaPipe similarity-transform → 112×112
    img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1

def expand_box(x1, y1, x2, y2, frame_h, frame_w):
    """Expand a tight face box to cover hair (top) and ears (sides)."""
    h, w = y2 - y1, x2 - x1
    x1 = max(0, int(x1 - 0.25 * w))
    x2 = min(frame_w, int(x2 + 0.25 * w))
    y1 = max(0, int(y1 - 0.45 * h))
    y2 = min(frame_h, int(y2 + 0.08 * h))
    return x1, y1, x2, y2


def apply_oval_to_frame(frame, x1, y1, x2, y2, obfuscated_region):
    """Paste obfuscated_region into frame as a filled rectangle."""
    rh, rw = y2 - y1, x2 - x1
    if rh <= 0 or rw <= 0:
        return
    frame[y1:y2, x1:x2] = cv2.resize(obfuscated_region, (rw, rh))


def apply_forbes_obfuscation(face_crop, brs_model, current_device):
    h, w = face_crop.shape[:2]
    if h == 0 or w == 0 or brs_model is None:
        return face_crop
        
    # Forbes explicitly expects 112x112 input
    img = cv2.resize(face_crop, (112, 112))
    img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    img_torch = img_torch.to(current_device)
    
    # BRS optimization needs gradients internally — do NOT wrap in torch.no_grad()
    output = brs_model.optimize(img_torch)
        
    # Convert [-1, 1] tensor back to BGR image array
    out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = (out_img + 1.0) * 127.5
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    
    obfuscated = cv2.resize(out_img, (w, h), interpolation=cv2.INTER_LINEAR)
    return obfuscated


def _pixelate(img_t, blocks=UAP_BLOCKS):
    """
    Strong pixelation via bilinear downscale → nearest-neighbor upscale.
    img_t: float tensor [1, 3, 112, 112] in [-1, 1].
    blocks: number of mosaic tiles per side (fewer = larger blocks = stronger privacy).
    """
    small = F.interpolate(img_t, size=(blocks, blocks), mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(112, 112), mode='nearest')


def apply_uap_obfuscation(face_crop, delta, current_device):
    """
    Real-time obfuscation: strong pixelation (visual layer) + UAP delta (adversarial layer).
    - Pixelation makes the face completely unrecognisable to human eyes.
    - UAP delta makes the face unrecognisable to AdaFace.
    """
    h, w = face_crop.shape[:2]
    if h == 0 or w == 0:
        return face_crop

    img = cv2.resize(face_crop, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    img_t = img_t.to(current_device)

    with torch.no_grad():
        # Visual layer: strong pixelation (UAP_BLOCKS tiles, ~22×22px each)
        mosaic = _pixelate(img_t)
        # Adversarial layer: universal perturbation on top of the pixelated image
        perturbed = torch.clamp(mosaic + delta, -1.0, 1.0)

    out = perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def obfuscate(face_crop):
    """Dispatch to UAP (fast) or Forbes (slow) depending on what's available."""
    if uap_delta is not None:
        return apply_uap_obfuscation(face_crop, uap_delta, device)
    return apply_forbes_obfuscation(face_crop, optimizer_brs, device)

# RTSP source 
RTSP_URL = "rtsp://admin:password@192.168.1.10:554/ch1/main"

# Shared slot holding the latest raw frame from the camera.
# A background thread writes here continuously so the processing loop always
# picks up the newest frame instead of a buffered one (which causes lag).
_cam_frame = None
_cam_lock  = threading.Lock()
_cam_running = True


def _camera_reader():
    """
    Dedicated thread that drains the RTSP buffer as fast as possible.
    Stores only the latest decoded frame so the processing loop is never
    working on stale data.  Reconnects automatically on stream errors.
    """
    global _cam_frame, _cam_running
    while _cam_running:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("Camera: could not connect — retrying in 3 s …")
            time.sleep(3)
            continue
        print("Camera: connected.")
        consecutive_fails = 0
        while _cam_running:
            ret, frame = cap.read()
            if ret:
                consecutive_fails = 0
                with _cam_lock:
                    _cam_frame = frame
            else:
                consecutive_fails += 1
                if consecutive_fails >= 10:
                    print("Camera: too many read errors — reconnecting …")
                    break
        cap.release()
        if _cam_running:
            time.sleep(2)


def main():
    global _cam_running 

    if optimizer_brs is None:
        print("Failed to initialize Forbes. Check your weights path. Exiting.")
        return

    # Start MJPEG stream server (Tab 2 in the dashboard connects to this)
    mjpeg_thread = threading.Thread(target=start_mjpeg_server, daemon=True)
    mjpeg_thread.start()
    print("MJPEG stream started at http://localhost:8001/video_feed")

    # Start the background camera reader — keeps the buffer drained
    cam_thread = threading.Thread(target=_camera_reader, daemon=True)
    cam_thread.start()

    # Pass the AdaFace model from Forbes into our Gallery
    gallery = FaceGallery(encoder_model=optimizer_brs.model, threshold=0.40)

    # Map temporary YOLO movement IDs to permanent Gallery IDs
    track_to_global_map = {}

    # Track how many frames a temporary YOLO ID has been visible
    track_history = {}

    # Track confirmed gallery person IDs seen this session for foot traffic
    all_seen_person_ids = set()

    # Smooth tracking state 
    # EMA-smoothed bounding box per track (floats): prevents the oval from
    # jumping due to per-frame detection noise.
    EMA_ALPHA = 0.4        # 0 = frozen, 1 = raw detection; 0.4 balances lag vs jitter
    # Per-frame velocity estimate (smoothed_box[t] - smoothed_box[t-1])
    # used to extrapolate position during missed detections.
    MAX_COAST = 6          # frames to keep obfuscating after YOLO loses the face

    track_smooth_box = {}  # {track_id: [x1,y1,x2,y2] float}
    track_velocity   = {}  # {track_id: [dx1,dy1,dx2,dy2] float}
    track_coast      = {}  # {track_id: int frames since last detection}

    # --- DB Setup ---
    db = AnalyticsDB()

    # Restore known identities so returning visitors are recognised across sessions
    gallery.load_from_db(db)
    last_log_time = time.time()
    LOG_INTERVAL = 2.0  # seconds between database writes

    print(f"Connecting to {RTSP_URL} …  Press Ctrl+C to quit.")

    try:
        while True:
            # Always grab the latest frame — skip if camera not ready yet
            with _cam_lock:
                if _cam_frame is None:
                    time.sleep(0.01)
                    continue
                frame = _cam_frame.copy()

            fh, fw = frame.shape[:2]

            # Run YOLO with built-in ByteTrack enabled.
            # conf=0.45 catches more detections during movement/profile angles.
            results = detector.track(frame, persist=True, conf=0.45, verbose=False)

            # Collect which track IDs YOLO actually detected this frame
            detected_ids = set()
            current_queue_length = 0

            for r in results:
                boxes = r.boxes
                if boxes.id is None:
                    continue

                track_ids = boxes.id.int().cpu().tolist()
                current_queue_length = max(current_queue_length, len(track_ids))

                for box, track_id in zip(boxes.xyxy, track_ids):
                    detected_ids.add(track_id)
                    raw = [float(v) for v in box]

                    # EMA smoothing 
                    if track_id in track_smooth_box:
                        prev = track_smooth_box[track_id]
                        track_velocity[track_id] = [raw[i] - prev[i] for i in range(4)]
                        smoothed = [EMA_ALPHA * raw[i] + (1 - EMA_ALPHA) * prev[i]
                                    for i in range(4)]
                    else:
                        smoothed = raw[:]
                        track_velocity[track_id] = [0.0, 0.0, 0.0, 0.0]
                    track_smooth_box[track_id] = smoothed
                    track_coast[track_id] = 0

                    x1, y1, x2, y2 = map(int, smoothed)
                    track_history[track_id] = track_history.get(track_id, 0) + 1

                    # Tight crop (original YOLO box) — for AdaFace identification.
                    # Face fills most of the 112×112 input, giving consistent embeddings.
                    tx1 = max(0, x1); ty1 = max(0, y1)
                    tx2 = min(fw, x2); ty2 = min(fh, y2)
                    tight_crop = frame[ty1:ty2, tx1:tx2]

                    # Expanded crop — for obfuscation overlay (covers hair and ears).
                    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, fh, fw)
                    face_crop = frame[ey1:ey2, ex1:ex2]
                    valid_crop = tight_crop.size > 0 and face_crop.size > 0

                    if track_id in track_to_global_map:
                        person_id = track_to_global_map[track_id]
                        label = f"Person {person_id}"
                        all_seen_person_ids.add(person_id)
                        if valid_crop:
                            apply_oval_to_frame(frame, ex1, ey1, ex2, ey2, obfuscate(face_crop))

                    elif track_history[track_id] > 5:
                        if valid_crop:
                            # Identify from the tight raw crop — face fills the 112×112
                            # frame so AdaFace gets the signal it was trained on.
                            face_tensor = preprocess_for_adaface(tight_crop)
                            person_id = gallery.match_or_create(face_tensor, db)
                            track_to_global_map[track_id] = person_id
                            all_seen_person_ids.add(person_id)
                            label = f"Person {person_id}"
                            # Obfuscate for display using the expanded crop
                            apply_oval_to_frame(frame, ex1, ey1, ex2, ey2, obfuscate(face_crop))
                        else:
                            continue
                    else:
                        label = "Tracking..."
                        if valid_crop:
                            apply_oval_to_frame(frame, ex1, ey1, ex2, ey2, obfuscate(face_crop))


            # Track coasting 
            # For every confirmed person whose face YOLO missed this frame,
            # extrapolate their box position using last known velocity and
            # continue obfuscating for up to MAX_COAST frames.
            for track_id, person_id in track_to_global_map.items():
                if track_id in detected_ids:
                    continue  # already handled above
                if track_id not in track_smooth_box:
                    continue
                track_coast[track_id] = track_coast.get(track_id, 0) + 1
                if track_coast[track_id] > MAX_COAST:
                    continue
                # Advance box by last velocity
                vel  = track_velocity.get(track_id, [0.0, 0.0, 0.0, 0.0])
                prev = track_smooth_box[track_id]
                coasted = [prev[i] + vel[i] for i in range(4)]
                track_smooth_box[track_id] = coasted
                cx1, cy1, cx2, cy2 = map(int, coasted)
                ex1, ey1, ex2, ey2 = expand_box(cx1, cy1, cx2, cy2, fh, fw)
                face_crop = frame[ey1:ey2, ex1:ex2]
                if face_crop.size > 0:
                    apply_oval_to_frame(frame, ex1, ey1, ex2, ey2, obfuscate(face_crop))

            # --- Calculate Analytics ---
            foot_traffic = len(all_seen_person_ids)
            unique_visitors = len(gallery.database)

            # --- Database Logging ---
            current_time = time.time()
            if current_time - last_log_time >= LOG_INTERVAL:
                db.log_metrics(current_queue_length, foot_traffic, unique_visitors)
                last_log_time = current_time

            # Push processed frame to MJPEG stream
            global output_frame
            with frame_lock:
                output_frame = frame.copy()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        _cam_running = False
        db.close()

if __name__ == "__main__":
    main()