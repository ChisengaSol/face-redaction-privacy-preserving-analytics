from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import sqlite3
import json
import cv2
import numpy as np
import threading
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

app = FastAPI(title="Privacy Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    conn = sqlite3.connect("analytics.db")
    conn.row_factory = sqlite3.Row
    return conn

#Analytics endpoints

@app.get("/api/analytics/latest")
def get_latest_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analytics_log ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {"message": "No data available"}

@app.get("/api/analytics/history")
def get_analytics_history(limit: int = 60):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analytics_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows][::-1]

@app.get("/api/embeddings")
def get_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM embeddings_log ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        row_dict = dict(row)
        row_dict['embedding'] = json.loads(row_dict['embedding'])
        result.append(row_dict)
    return result

@app.get("/api/analytics/summary")
def get_analytics_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(queue_length) AS peak_queue, CAST(AVG(queue_length) AS REAL) AS avg_queue,
               MAX(foot_traffic) AS total_foot_traffic, MAX(unique_visitors) AS total_unique
        FROM analytics_log WHERE timestamp >= datetime('now', '-24 hours')
    """)
    row = cursor.fetchone()
    conn.close()
    if row and row["peak_queue"] is not None:
        return {"peak_queue": row["peak_queue"], "avg_queue": round(row["avg_queue"] or 0, 1),
                "total_foot_traffic": row["total_foot_traffic"] or 0, "total_unique": row["total_unique"] or 0}
    return {"peak_queue": 0, "avg_queue": 0.0, "total_foot_traffic": 0, "total_unique": 0}

@app.get("/api/analytics/hourly")
def get_hourly_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strftime('%H', timestamp) AS hour, MAX(queue_length) AS peak_queue,
               CAST(AVG(queue_length) AS REAL) AS avg_queue
        FROM analytics_log WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY hour ORDER BY hour
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"hour": f"{r['hour']}:00", "peak_queue": r["peak_queue"],
             "avg_queue": round(r["avg_queue"], 1)} for r in rows]

# Video processing helpers 

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_detector_lock = threading.Lock()
_yolo_detector = None

def _get_detector():
    global _yolo_detector
    with _detector_lock:
        if _yolo_detector is None:
            from ultralytics import YOLO
            _yolo_detector = YOLO("../yolo_detector/weights/best.pt")
    return _yolo_detector

_adaface_lock = threading.Lock()
_adaface_model = None
_adaface_device = None

def _get_adaface():
    """Lazy-load AdaFace. Returns (model, device) or None if weights not found."""
    global _adaface_model, _adaface_device
    with _adaface_lock:
        if _adaface_model is None:
            try:
                import torch
                import forbes
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                brs = forbes.init_forbes("adaface_ir101_ms1mv2.ckpt", device)
                if brs is not None:
                    _adaface_model = brs.model
                    _adaface_device = device
                    print("AdaFace loaded for video re-ID.")
                else:
                    print("AdaFace init failed — video will have no re-ID.")
            except Exception as e:
                print(f"AdaFace load error: {e} — video will have no re-ID.")
    return (_adaface_model, _adaface_device) if _adaface_model else None


class VideoFaceGallery:
    """
    Re-ID gallery for offline video processing.

    Accumulates ACCUMULATE_N embeddings per new track before committing.
    Averaging over multiple frames removes single-frame noise that causes
    wrong cross-person matches when using a low threshold.
    """
    ACCUMULATE_N = 12   # frames to average before matching/registering
    THRESHOLD    = 0.40  # cosine similarity cutoff (higher = fewer false merges)

    def __init__(self, encoder, device):
        self.encoder  = encoder
        self.device   = device
        self.database = {}   # {person_id: avg_embedding_tensor}
        self.next_id  = 1
        self.pending  = {}   # {track_id: [embedding, ...]} accumulation buffer

    def feed(self, track_id, face_tensor):
        """
        Feed one frame's embedding for a track.
        Returns a person_id once enough frames have been accumulated,
        otherwise returns None (caller should show 'Tracking...').
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            emb, _ = self.encoder(face_tensor.to(self.device))

        if track_id not in self.pending:
            self.pending[track_id] = []
        self.pending[track_id].append(emb)

        if len(self.pending[track_id]) < self.ACCUMULATE_N:
            return None   # still collecting

        # Average accumulated embeddings → stable representative vector
        stack   = torch.cat(self.pending.pop(track_id), dim=0)   # [N, 512]
        avg_emb = F.normalize(stack.mean(dim=0, keepdim=True), dim=1)

        return self._match_or_register(track_id, avg_emb)

    def _match_or_register(self, track_id, embedding):
        import torch.nn.functional as F

        best_id, max_sim = None, -1.0
        all_sims = {}
        for pid, stored in self.database.items():
            sim = F.cosine_similarity(embedding, stored).item()
            all_sims[pid] = sim
            if sim > max_sim:
                max_sim, best_id = sim, pid

        sim_str = "  ".join(f"P{p}={s:.3f}" for p, s in sorted(all_sims.items()))
        print(f"[Video ReID] track={track_id}  sims=[{sim_str}]  best={max_sim:.3f}  thresh={self.THRESHOLD}")

        if max_sim > self.THRESHOLD:
            # EMA update with the new averaged embedding
            updated = 0.85 * self.database[best_id] + 0.15 * embedding
            import torch.nn.functional as F
            self.database[best_id] = F.normalize(updated, dim=1)
            print(f"[Video ReID] → Matched Person {best_id}")
            return best_id

        new_id = self.next_id
        self.database[new_id] = embedding
        self.next_id += 1
        print(f"[Video ReID] → New Person {new_id}")
        return new_id


def _preprocess_for_adaface(face_img):
    from face_align import align_face
    aligned = align_face(face_img)   # MediaPipe similarity-transform → 112×112
    img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    import torch
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1

def _expand_box(x1, y1, x2, y2, h, w):
    bh, bw = y2 - y1, x2 - x1
    return (max(0, int(x1 - 0.25*bw)), max(0, int(y1 - 0.45*bh)),
            min(w, int(x2 + 0.25*bw)), min(h, int(y2 + 0.08*bh)))

def _strong_pixelate(frame, x1, y1, x2, y2, blocks=5):
    """Resize down then up — strong square mosaic, no mask."""
    rh, rw = y2 - y1, x2 - x1
    if rh <= 0 or rw <= 0:
        return
    roi   = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

def _find_ffmpeg() -> str | None:
    """
    Locate an ffmpeg binary from multiple sources:
    1. System PATH
    2. imageio-ffmpeg package (ships its own static binary — most reliable on Windows)
    """
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return None


def _encode_h264(input_path: str, output_path: str) -> bool:
    """Re-encode to H.264 with ffmpeg for browser playback. Returns True on success."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg not found. Install imageio-ffmpeg:  pip install imageio-ffmpeg")
        return False
    print(f"ffmpeg found at: {ffmpeg}")
    try:
        result = subprocess.run(
            [ffmpeg, "-i", input_path,
             "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
             "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-y", output_path],
            capture_output=True, timeout=600
        )
        if result.returncode != 0:
            print("ffmpeg stderr:", result.stderr.decode(errors="ignore")[-500:])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

# Video state 

video_state = {
    "status":           "idle",
    "progress":         0.0,
    "total_frames":     0,
    "processed_frames": 0,
    "output_filename":  None,
    "error":            None,
    "analytics":        [],   # [{time, queue, unique}, …] sampled once per second
    "avg_dwell_sec":    0,
}
_state_lock = threading.Lock()


def _process_video(input_path: str, output_filename: str):
    try:
        import torch
        detector = _get_detector()
        adaface  = _get_adaface()

        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_path   = str(UPLOAD_DIR / f"_tmp_{output_filename}")
        output_path = str(UPLOAD_DIR / output_filename)
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Re-ID gallery — threshold 0.25 to handle real-world face quality
        gallery         = VideoFaceGallery(adaface[0], adaface[1]) if adaface else None
        track_to_person = {}   # YOLO track_id → gallery person_id
        track_frames    = {}   # track_id → consecutive frames seen
        all_seen        = set()
        person_first_seen_sec = {}   # {person_id: float}
        person_last_seen_sec  = {}   # {person_id: float}

        # EMA smoothing + coasting state 
        EMA_ALPHA      = 0.4   # blend factor for box smoothing
        MAX_COAST      = 8     # frames to keep obfuscating after detection drops
        smooth_box     = {}    # {track_id: [x1,y1,x2,y2] float}
        velocity       = {}    # {track_id: [dx1,dy1,dx2,dy2] float}
        coast_counter  = {}    # {track_id: frames since last detection}
        

        analytics    = []
        sample_every = max(1, int(fps))

        with _state_lock:
            video_state.update({
                "status": "processing", "total_frames": total,
                "processed_frames": 0, "analytics": []
            })

        idx  = 0
        pbar = tqdm(total=total, desc="Processing", unit="frame",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            queue_length = 0
            detected_ids = set()

            # Lower conf=0.4 catches more detections during movement/side angles
            results = detector.track(frame, persist=True, conf=0.4, verbose=False)

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                track_ids    = r.boxes.id.int().cpu().tolist()
                queue_length = max(queue_length, len(track_ids))

                for box, track_id in zip(r.boxes.xyxy, track_ids):
                    detected_ids.add(track_id)
                    raw = [float(v) for v in box]

                    # EMA smoothing 
                    if track_id in smooth_box:
                        prev = smooth_box[track_id]
                        velocity[track_id]  = [raw[i] - prev[i] for i in range(4)]
                        smoothed = [EMA_ALPHA*raw[i] + (1-EMA_ALPHA)*prev[i] for i in range(4)]
                    else:
                        smoothed = raw[:]
                        velocity[track_id]  = [0.0]*4
                    smooth_box[track_id]   = smoothed
                    coast_counter[track_id] = 0

                    x1, y1, x2, y2 = map(int, smoothed)
                    track_frames[track_id] = track_frames.get(track_id, 0) + 1

                    # Tight crop → AdaFace (face fills 112×112)
                    tx1, ty1 = max(0, x1), max(0, y1)
                    tx2, ty2 = min(width, x2), min(height, y2)
                    tight = frame[ty1:ty2, tx1:tx2]

                    # Expanded crop → obfuscation overlay
                    ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, height, width)

                    # Feed each frame's embedding into the accumulation buffer.
                    # gallery.feed() returns a person_id only after ACCUMULATE_N
                    # frames have been averaged — avoids single-frame false matches.
                    if track_id not in track_to_person and gallery is not None and tight.size > 0:
                        face_t = _preprocess_for_adaface(tight)
                        pid    = gallery.feed(track_id, face_t)
                        if pid is not None:
                            track_to_person[track_id] = pid

                    person_id = track_to_person.get(track_id)
                    if person_id:
                        all_seen.add(person_id)
                        t_sec = idx / fps
                        if person_id not in person_first_seen_sec:
                            person_first_seen_sec[person_id] = t_sec
                        person_last_seen_sec[person_id] = t_sec

                    _strong_pixelate(frame, ex1, ey1, ex2, ey2)

            # Track coasting 
            # For confirmed persons YOLO missed this frame, extrapolate position
            # and keep the obfuscation on them for up to MAX_COAST frames.
            for track_id, person_id in track_to_person.items():
                if track_id in detected_ids:
                    continue
                if track_id not in smooth_box:
                    continue
                coast_counter[track_id] = coast_counter.get(track_id, 0) + 1
                if coast_counter[track_id] > MAX_COAST:
                    continue
                vel  = velocity.get(track_id, [0.0]*4)
                prev = smooth_box[track_id]
                coasted = [prev[i] + vel[i] for i in range(4)]
                smooth_box[track_id] = coasted
                cx1, cy1, cx2, cy2 = map(int, coasted)
                ex1, ey1, ex2, ey2 = _expand_box(cx1, cy1, cx2, cy2, height, width)
                _strong_pixelate(frame, ex1, ey1, ex2, ey2)

            if idx % sample_every == 0:
                analytics.append({
                    "time":   round(idx / fps, 1),
                    "queue":  queue_length,
                    "unique": len(all_seen),
                })

            out.write(frame)
            idx += 1
            pbar.update(1)
            with _state_lock:
                video_state["processed_frames"] = idx
                video_state["progress"] = round(idx / total * 100, 1) if total else 0

        pbar.close()
        cap.release()
        out.release()

        # Re-encode to H.264 so browsers can play it
        print("Re-encoding to H.264 for browser playback …")
        if _encode_h264(temp_path, output_path):
            Path(temp_path).unlink(missing_ok=True)
            print("H.264 encoding done.")
        else:
            # ffmpeg not found — serve the mp4v file as-is (may not play in all browsers)
            print("ffmpeg not found — serving raw mp4v file.")
            shutil.move(temp_path, output_path)

        dwell_times = [
            person_last_seen_sec[pid] - person_first_seen_sec[pid]
            for pid in all_seen
            if pid in person_first_seen_sec and pid in person_last_seen_sec
        ]
        avg_dwell_sec = round(sum(dwell_times) / len(dwell_times), 1) if dwell_times else 0

        with _state_lock:
            video_state.update({
                "status":          "done",
                "progress":        100.0,
                "output_filename": output_filename,
                "analytics":       analytics,
                "avg_dwell_sec":   avg_dwell_sec,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _state_lock:
            video_state.update({"status": "error", "error": str(e)})


# Video endpoints 

@app.post("/api/video/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    with _state_lock:
        video_state.update({"status": "uploading", "progress": 0,
                             "output_filename": None, "error": None, "analytics": []})

    ext             = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path      = str(UPLOAD_DIR / f"input{ext}")
    output_filename = f"output.mp4"   # always mp4 output

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_process_video, input_path, output_filename)
    return {"message": "Processing started", "filename": file.filename}


@app.get("/api/video/status")
def get_video_status():
    with _state_lock:
        state = dict(video_state)
        state.pop("analytics", None)   # keep status response lightweight
    return state


@app.get("/api/video/analytics")
def get_video_analytics():
    with _state_lock:
        return {
            "data":          video_state.get("analytics", []),
            "avg_dwell_sec": video_state.get("avg_dwell_sec", 0),
        }


@app.get("/api/video/output")
def get_processed_video(request: Request):
    with _state_lock:
        status   = video_state["status"]
        filename = video_state["output_filename"]

    if status != "done" or not filename:
        return {"error": "Video not ready yet"}

    path = UPLOAD_DIR / filename
    if not path.exists():
        return {"error": "Output file not found"}

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    # Support HTTP range requests so the browser can seek in the video
    if range_header:
        range_val  = range_header.replace("bytes=", "").split("-")
        start      = int(range_val[0])
        end        = int(range_val[1]) if range_val[1] else file_size - 1
        chunk_size = end - start + 1

        def iter_file(s, e):
            with open(path, "rb") as f:
                f.seek(s)
                remaining = e - s + 1
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(start, end),
            status_code=206,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type":   "video/mp4",
            },
        )

    return FileResponse(str(path), media_type="video/mp4",
                        filename="privacy_processed.mp4",
                        headers={"Accept-Ranges": "bytes"})
