"""
diagnose_embeddings.py  —  Embedding quality report

Run from inside the app/ folder:
    python diagnose_embeddings.py

Reads every stored embedding from analytics.db and tells you:
  - How many identities exist
  - The full pairwise cosine similarity matrix
  - Whether the current threshold (0.45) makes sense for your data
"""

import json
import sqlite3
import numpy as np

DB_PATH   = "analytics.db"
THRESHOLD = 0.45   # must match FaceGallery threshold in app.py

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT person_id, embedding, timestamp FROM embeddings_log ORDER BY person_id")
rows = cursor.fetchall()
conn.close()

if not rows:
    print("No embeddings found in the database. Run app.py first so it can detect and store faces.")
    exit()

ids        = [r[0] for r in rows]
embeddings = [np.array(json.loads(r[1]), dtype=np.float32) for r in rows]
timestamps = [r[2] for r in rows]

# Normalise (they should already be unit vectors from AdaFace, but just in case)
embeddings = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]

n = len(ids)
print(f"\n{'='*60}")
print(f"  Embedding Diagnostic Report")
print(f"{'='*60}")
print(f"  Identities stored : {n}")
print(f"  Threshold in use  : {THRESHOLD}")
print()

print("  Pairwise Cosine Similarity Matrix")
print("  (rows/cols = Person IDs, value = similarity  1.0=identical  0.0=unrelated)")
print()

# Header row
header = "         " + "".join(f"  P{pid:<4}" for pid in ids)
print(header)
print("  " + "-" * (len(header) - 2))

sims = np.zeros((n, n))
for i in range(n):
    row_str = f"  P{ids[i]:<5} |"
    for j in range(n):
        s = float(np.dot(embeddings[i], embeddings[j]))
        sims[i, j] = s
        if i == j:
            row_str += "   ---  "
        elif s >= THRESHOLD:
            row_str += f" [{s:+.3f}]"   # bracket = would be matched
        else:
            row_str += f"  {s:+.3f} "
    print(row_str)

print()
print("  [ value ] = above threshold → would be matched as the same person")
print()

if n > 1:
    upper = [(sims[i, j], ids[i], ids[j])
             for i in range(n) for j in range(i+1, n)]
    values = [v for v, _, _ in upper]

    print(f"  Off-diagonal statistics ({len(values)} pairs):")
    print(f"    Min similarity  : {min(values):+.4f}")
    print(f"    Max similarity  : {max(values):+.4f}")
    print(f"    Mean similarity : {np.mean(values):+.4f}")
    print(f"    Std deviation   : {np.std(values):+.4f}")
    print()

    above = [(v, a, b) for v, a, b in upper if v >= THRESHOLD]
    print(f"  Pairs that EXCEED threshold {THRESHOLD} (risk of two people sharing an ID):")
    if above:
        for v, a, b in sorted(above, reverse=True):
            print(f"    Person {a} ↔ Person {b}  →  {v:+.4f}")
    else:
        print("    None — all stored identities are distinct from each other.")
    print()

print("  Interpretation guide:")
print("  ┌─────────────────────────────────────────────────────────┐")
print("  │ Same-person similarity (what we want for re-ID):        │")
print("  │   > 0.70  Excellent — re-ID will be reliable            │")
print("  │   0.45–0.70  Acceptable — threshold is in safe zone     │")
print("  │   < 0.45  Poor — same person will get a new ID each time│")
print("  │                                                         │")
print("  │ Different-person similarity (what we want for privacy): │")
print("  │   < 0.30  Excellent — identities are well separated     │")
print("  │   0.30–0.45  Acceptable — small overlap with threshold  │")
print("  │   > 0.45  Problem — different people merged into one ID │")
print("  └─────────────────────────────────────────────────────────┘")
print()

print("  Raw embedding norms (should all be ≈ 1.0 if AdaFace is working):")
for pid, emb in zip(ids, embeddings):
    norm = float(np.linalg.norm(emb))
    print(f"    Person {pid:<4}: norm = {norm:.6f}  (stored: {timestamps[ids.index(pid)]})")

print()
print(f"{'='*60}\n")
