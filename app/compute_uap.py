"""
compute_uap.py  —  Universal Adversarial Perturbation calibration script

Run this ONCE before starting the main app:
    python compute_uap.py

It opens your webcam, collects face crops, then runs PGD to find a single
perturbation delta that fools AdaFace on all collected faces simultaneously.
The result is saved to uap_delta.npy.

app.py will automatically detect and load this file at startup.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import forbes

#Config 
EPSILON      = 0.10   # max L∞ perturbation in [-1,1] space (~13 pixel values)
STEPS        = 50     # PGD iterations
ALPHA        = 0.004  # step size per iteration (EPSILON / ~25)
N_FACES      = 40     # face crops to collect for calibration
BATCH_SIZE   = 8      # mini-batch size during optimisation
MIN_INTERVAL = 0.5    # min seconds between collections (ensures variety)

UAP_PATH     = "uap_delta.npy"
ADAFACE_CKPT = "adaface_ir101_ms1mv2.ckpt"
# MUST match app.py UAP_BLOCKS — UAP is optimised in the pixelated domain,
# so it must be applied to the same pixelation at runtime.
UAP_BLOCKS   = 5      # mosaic tiles per side on 112×112 → ~22×22px blocks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pixelate(img_t, blocks=UAP_BLOCKS):
    """Downscale then upscale to create strong mosaic tiles."""
    small = F.interpolate(img_t, size=(blocks, blocks), mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(112, 112), mode='nearest')


# Step 1: collect face crops from webcam 
def collect_faces(detector, n=N_FACES):
    cap        = cv2.VideoCapture(0)
    faces      = []
    last_time  = 0.0

    print(f"\n[1/2] Point the camera at faces — collecting {n} crops.")
    print("      Move around, change angles and distances for variety.")
    print("      Press Q to stop early.\n")

    while len(faces) < n:
        ret, frame = cap.read()
        if not ret:
            continue

        results = detector(frame, conf=0.6, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                # Skip tiny or empty detections
                if crop.size == 0 or (y2 - y1) < 40 or (x2 - x1) < 40:
                    continue

                # Rate-limit to ensure variety across poses / lighting
                now = time.time()
                if now - last_time < MIN_INTERVAL:
                    continue
                last_time = now

                # Preprocess exactly as AdaFace expects: RGB, 112×112, [-1,1]
                img    = cv2.resize(crop, (112, 112))
                img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
                faces.append(tensor)
                print(f"  Collected {len(faces)}/{n}", end='\r')

                if len(faces) >= n:
                    break

        # Live feedback overlay
        cv2.putText(frame, f"Faces collected: {len(faces)}/{n}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("UAP Calibration  —  press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not faces:
        raise RuntimeError("No faces collected. Make sure YOLO can detect faces from your camera.")

    print(f"\n  Done — collected {len(faces)} crops.")
    return torch.stack(faces)   # [N, 3, 112, 112]


# Step 2: PGD optimisation 
def compute_uap(model, faces):
    faces = faces.to(device)
    model.eval()

    # Pixelate all collected faces — the UAP must be optimised in the same
    # domain it will be applied to at runtime (pixelated, not raw).
    with torch.no_grad():
        faces_px = _pixelate(faces)

    # Cache embeddings of the PIXELATED faces as the targets to move away from.
    with torch.no_grad():
        parts = [model(faces_px[i:i+BATCH_SIZE])[0]
                 for i in range(0, len(faces_px), BATCH_SIZE)]
        orig_embs = torch.cat(parts, dim=0)   # [N, 512]

    # Universal delta — shape [1, 3, 112, 112] broadcasts across all faces
    delta = torch.zeros(1, 3, 112, 112, device=device, requires_grad=True)

    print(f"\n[2/2] Running PGD  ({STEPS} steps, ε={EPSILON})\n")
    pbar = tqdm(range(STEPS), unit="step")
    pbar.set_postfix(sim=1.0)

    for _ in pbar:
        if delta.grad is not None:
            delta.grad.zero_()

        total_sim, n_batches = 0.0, 0

        for i in range(0, len(faces_px), BATCH_SIZE):
            batch      = faces_px[i:i + BATCH_SIZE]
            batch_orig = orig_embs[i:i + BATCH_SIZE]

            # Add delta to the pixelated faces (same as runtime)
            perturbed        = torch.clamp(batch + delta, -1.0, 1.0)
            emb_perturbed, _ = model(perturbed)

            # Minimise cosine similarity → maximise embedding distance
            loss = F.cosine_similarity(batch_orig.detach(), emb_perturbed).mean()
            loss.backward()

            total_sim += loss.item()
            n_batches += 1

        # PGD update: signed gradient descent + L∞ projection
        with torch.no_grad():
            delta.data -= ALPHA * delta.grad.sign()
            delta.data  = delta.data.clamp(-EPSILON, EPSILON)

        pbar.set_postfix(sim=f"{total_sim / n_batches:.4f}")

    final_sim = total_sim / n_batches
    print(f"\n  Final average cosine similarity: {final_sim:.4f}")
    print(f"  (1.0 = identical embeddings  |  0.0 = orthogonal  |  -1.0 = maximally different)")

    return delta.detach()


def main():
    print("Loading YOLO face detector...")
    detector = YOLO("../yolo_detector/weights/best.pt")

    print("Loading AdaFace model...")
    brs = forbes.init_forbes(ADAFACE_CKPT, device)
    if brs is None:
        print("ERROR: Could not load AdaFace. Check ADAFACE_CKPT path.")
        return
    model = brs.model

    faces = collect_faces(detector)
    delta = compute_uap(model, faces)

    np.save(UAP_PATH, delta.cpu().numpy())
    print(f"\nSaved UAP delta to '{UAP_PATH}'")
    print("Run app.py — it will automatically use UAP instead of Forbes.\n")


if __name__ == "__main__":
    main()
