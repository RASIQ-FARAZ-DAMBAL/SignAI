"""
export_poses.py
---------------
Picks the most representative (centroid-closest) landmark sample
for each letter and exports static/sign_poses.json for the animation.

Run from your signbridge/ folder:
    python export_poses.py
"""

import os
import json
import numpy as np

DATA_DIR  = "pretrained_data/real_webcam"
OUT_FILE  = "static/sign_poses.json"

poses = {}

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    samples = []
    for f in os.listdir(label_path):
        if f.endswith(".npy"):
            samples.append(np.load(os.path.join(label_path, f)))

    if not samples:
        print(f"  SKIP {label} — no samples")
        continue

    samples = np.array(samples)          # (N, 63)
    centroid = samples.mean(axis=0)      # average pose

    # pick the actual sample closest to the centroid (cleanest representative)
    dists = np.linalg.norm(samples - centroid, axis=1)
    best  = samples[np.argmin(dists)]    # (63,)

    # reshape to list of {x, y, z} dicts — easier to use in JS
    coords = best.reshape(21, 3)
    poses[label] = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in coords]

    print(f"  OK  {label}  ({len(samples)} samples)")

os.makedirs("static", exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(poses, f)

print(f"\nExported {len(poses)} letter poses → {OUT_FILE}")