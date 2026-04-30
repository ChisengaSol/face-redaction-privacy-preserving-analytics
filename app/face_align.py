"""
face_align.py — MediaPipe-based face alignment for AdaFace.

AdaFace was trained on faces aligned to a canonical 112×112 template where
both eyes are at fixed y=51, nose at centre, and mouth at y=92.  Feeding an
unaligned crop produces inconsistent embeddings — same person looks completely
different each frame — making re-ID unreliable.

align_face() detects iris + mouth landmarks with MediaPipe FaceLandmarker
(Tasks API, compatible with mediapipe ≥ 0.10.30), then applies a similarity
transform (rotation + uniform scale + translation, no shear) to warp the crop
into that canonical space.  Falls back to plain resize if landmarks cannot be
found (face too small, extreme angle, model unavailable, etc.).
"""

import os
import threading
import urllib.request
import numpy as np
import cv2

# ─── AdaFace / ArcFace canonical 5-point template (output size 112 × 112) ────
# Order: left-eye, right-eye, nose-tip, left-mouth-corner, right-mouth-corner
# "left/right" are from the CAMERA's perspective (mirror of the person's).
_TEMPLATE = np.float32([
    [38.29, 51.70],   # camera-left  eye
    [73.53, 51.50],   # camera-right eye
    [56.02, 71.74],   # nose tip
    [41.55, 92.37],   # camera-left  mouth corner
    [70.73, 92.20],   # camera-right mouth corner
])

# MediaPipe FaceLandmarker model — downloaded automatically on first use.
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")

_landmarker      = None
_landmarker_lock = threading.Lock()


def _ensure_model() -> bool:
    """Download the model file if it isn't already on disk. Returns True on success."""
    if os.path.exists(_MODEL_PATH):
        return True
    try:
        print("[Align] Downloading face landmarker model (~29 MB) — one-time setup …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[Align] Model saved to {_MODEL_PATH}")
        return True
    except Exception as exc:
        print(f"[Align] Model download failed: {exc}")
        return False


def _get_landmarker():
    """Lazy-load a single shared FaceLandmarker instance (new Tasks API)."""
    global _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision as mp_vision

                if not _ensure_model():
                    _landmarker = "unavailable"
                else:
                    base = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
                    opts = mp_vision.FaceLandmarkerOptions(
                        base_options=base,
                        num_faces=1,
                        min_face_detection_confidence=0.3,
                        min_face_presence_confidence=0.3,
                        min_tracking_confidence=0.3,
                    )
                    _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
                    print("[Align] MediaPipe FaceLandmarker loaded for alignment.")
            except Exception as exc:
                print(f"[Align] MediaPipe setup failed: {exc}")
                print("[Align] Falling back to plain resize — re-ID accuracy reduced.")
                _landmarker = "unavailable"

    return _landmarker if _landmarker != "unavailable" else None


def align_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Align a BGR face crop to the AdaFace 112×112 canonical format.

    Parameters
    ----------
    face_bgr : np.ndarray
        Tight face crop from the YOLO bounding box (any size, BGR).

    Returns
    -------
    np.ndarray
        112×112 BGR image aligned to the AdaFace template, or plain-resized
        fallback if MediaPipe cannot find landmarks.
    """
    h, w = face_bgr.shape[:2]
    fallback = cv2.resize(face_bgr, (112, 112))

    if h < 24 or w < 24:
        return fallback

    lmk = _get_landmarker()
    if lmk is None:
        return fallback

    try:
        import mediapipe as mp
        rgb      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = lmk.detect(mp_image)

        if not result.face_landmarks:
            return fallback

        lm = result.face_landmarks[0]   # list of NormalizedLandmark

        # FaceLandmarker iris indices (same as old refine_landmarks=True):
        #   468 = left  iris centre (person's left  = camera's right = larger x)
        #   473 = right iris centre (person's right = camera's left  = smaller x)
        # Template order: [camera-left eye, camera-right eye, nose, mouth-L, mouth-R]
        src = np.float32([
            [lm[473].x * w, lm[473].y * h],   # camera-left  eye
            [lm[468].x * w, lm[468].y * h],   # camera-right eye
            [lm[1  ].x * w, lm[1  ].y * h],   # nose tip
            [lm[61 ].x * w, lm[61 ].y * h],   # camera-left  mouth corner
            [lm[291].x * w, lm[291].y * h],   # camera-right mouth corner
        ])

        M, _ = cv2.estimateAffinePartial2D(src, _TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return fallback

        return cv2.warpAffine(
            face_bgr, M, (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    except Exception:
        return fallback
