import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = "pretrained_data/real_webcam"

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
# Raw landmarks alone (63 values) aren't enough to separate similar signs.
# We add:
#   1. Finger angles       — angle at each knuckle joint (PIP bend)
#   2. Inter-tip distances — distances between all fingertip pairs
#   3. Tip-to-wrist distances — how far each fingertip is from wrist
# This gives the model geometric context that raw x/y/z can't provide alone.

FINGERTIP_IDS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
FINGER_MID_IDS = [3, 7, 11, 15, 19]   # PIP joints (middle knuckles)
FINGER_BASE_IDS= [2, 6, 10, 14, 18]   # MCP joints (base knuckles)

def angle_between(a, b, c):
    """Angle at point b formed by vectors b->a and b->c (in degrees)."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def extract_features(raw):
    """
    raw: flat array of 63 values (21 landmarks × 3 coords), already
         wrist-normalised by collect_data.py
    Returns: enriched feature vector
    """
    coords = raw.reshape(21, 3)

    # 1. raw normalised landmarks (63)
    features = list(raw)

    # 2. finger bend angles at PIP joints (5)
    for tip, mid, base in zip(FINGERTIP_IDS, FINGER_MID_IDS, FINGER_BASE_IDS):
        ang = angle_between(coords[tip], coords[mid], coords[base])
        features.append(ang / 180.0)   # normalise to [0,1]

    # 3. inter-fingertip distances — all pairs (10)
    for i in range(len(FINGERTIP_IDS)):
        for j in range(i + 1, len(FINGERTIP_IDS)):
            d = np.linalg.norm(coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]])
            features.append(d)

    # 4. each fingertip distance from wrist (landmark 0) (5)
    for tip in FINGERTIP_IDS:
        d = np.linalg.norm(coords[tip] - coords[0])
        features.append(d)

    return np.array(features)   # 63 + 5 + 10 + 5 = 83 features


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
X, y = [], []

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            raw = np.load(os.path.join(label_path, file))
            X.append(extract_features(raw))
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Dataset loaded: {X.shape}  ({len(set(y))} classes)")

# ── SPLIT ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── MODEL — SVM with RBF kernel ───────────────────────────────────────────────
# SVM + RBF consistently outperforms Random Forest on small datasets with
# similar high-dimensional classes (exactly our M/N, R/U problem).
# StandardScaler is mandatory before SVM.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,            # regularisation — higher = tighter fit
        gamma="scale",   # auto-scales with feature variance
        probability=True # needed for predict_proba in app.py
    ))
])

print("Training SVM...")
model.fit(X_train, y_train)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc * 100:.2f}%")

# Show per-class report — focus on M/N/R/U
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix for the known problem pairs
CONFUSED_PAIRS = ["M", "N", "R", "U"]
pair_labels = [l for l in CONFUSED_PAIRS if l in set(y)]
if len(pair_labels) > 1:
    mask = np.isin(y_test, pair_labels)
    if mask.sum() > 0:
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=pair_labels)
        print(f"\nConfusion matrix for {pair_labels}:")
        print("Predicted →  " + "  ".join(f"{l:>4}" for l in pair_labels))
        for i, row in enumerate(cm):
            print(f"Actual {pair_labels[i]:>4}:  " + "  ".join(f"{v:>4}" for v in row))

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sign_model.pkl")
print("\nModel saved to models/sign_model.pkl")
print("NOTE: app.py will work as-is — Pipeline handles scaling automatically.")