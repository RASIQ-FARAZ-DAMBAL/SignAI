import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import argparse

# ================= SETTINGS =================
ALL_SIGNS       = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("123456789") + ["SPACE", "BACKSPACE", "CLEAR"]
SAMPLES_PER_SIGN = 600     # increased from 500 for better coverage
SAVE_DIR        = "pretrained_data/real_webcam"
CAPTURE_DELAY   = 0.04     # 40ms — slightly faster capture

# Signs most commonly confused — collect extra for these
PRIORITY_SIGNS  = ["M", "N", "R", "U", "V", "T", "K", "D"]

# Gesture signs — special hand poses for controls
GESTURE_SIGNS = {
    "SPACE":     "Open palm — all 5 fingers fully stretched out",
    "BACKSPACE": "Thumbs down — fist with thumb pointing downward",
    "CLEAR":     "Closed fist — all fingers curled in tightly",
}

# ===========================================

parser = argparse.ArgumentParser(description="SignAI Data Collector")
parser.add_argument(
    "--signs", nargs="+", default=None,
    help="Specific signs to collect e.g. --signs M N R U (default: all)"
)
parser.add_argument(
    "--samples", type=int, default=SAMPLES_PER_SIGN,
    help=f"Samples per sign (default: {SAMPLES_PER_SIGN})"
)
parser.add_argument(
    "--priority", action="store_true",
    help="Only collect priority/confused signs: " + " ".join(PRIORITY_SIGNS)
)
parser.add_argument(
    "--append", action="store_true",
    help="Append to existing data instead of overwriting"
)
args = parser.parse_args()

# Determine which signs to collect
if args.priority:
    SIGNS = PRIORITY_SIGNS
elif args.signs:
    SIGNS = [s.upper() for s in args.signs]
else:
    SIGNS = ALL_SIGNS

SAMPLES = args.samples
print(f"\n📋 Signs to collect : {SIGNS}")
print(f"📊 Samples per sign : {SAMPLES}")
print(f"📁 Save directory   : {SAVE_DIR}")
print(f"➕ Append mode      : {args.append}\n")

# ===========================================

BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6
)

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

# ---------- NORMALIZATION (same as app.py) ----------
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords = coords - coords[0]
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val
    return coords.flatten()

# ---------- GET NEXT FILE INDEX ----------
def next_index(sign_dir, append_mode):
    if not append_mode:
        # wipe existing
        import shutil
        if os.path.exists(sign_dir):
            shutil.rmtree(sign_dir)
        os.makedirs(sign_dir, exist_ok=True)
        return 0
    else:
        os.makedirs(sign_dir, exist_ok=True)
        existing = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        return len(existing)

# -----------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:
    for sign_idx, sign in enumerate(SIGNS):
        sign_dir   = os.path.join(SAVE_DIR, sign)
        start_idx  = next_index(sign_dir, args.append)
        is_priority = sign in PRIORITY_SIGNS
        is_gesture  = sign in GESTURE_SIGNS

        print(f"\n{'⭐' if is_priority else '📸'} [{sign_idx+1}/{len(SIGNS)}] Sign: '{sign}'"
              f"  (existing: {start_idx}, collecting: {SAMPLES})")
        if is_gesture:
            print(f"  🤟 Gesture: {GESTURE_SIGNS[sign]}")
        print("Press SPACE to start  |  ESC to skip  |  Q to quit")

        # WAIT SCREEN
        skipped = False
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            label = f"READY: {sign}"
            if is_priority:
                label += "  [PRIORITY - confused sign]"
            if is_gesture:
                label += "  [GESTURE]"
            cv2.putText(frame, label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 2)
            if is_gesture:
                cv2.putText(frame, GESTURE_SIGNS[sign], (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 1)
            cv2.putText(frame, f"Existing samples: {start_idx}", (30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(frame, "SPACE = start | ESC = skip | Q = quit", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            cv2.imshow("SignAI Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:  # ESC
                skipped = True
                break
            elif key == ord('q'):
                print("\n🛑 Quit by user.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if skipped:
            print(f"  ⏭  Skipped {sign}")
            continue

        count     = 0
        last_time = 0

        while count < SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)
            result    = landmarker.detect_for_video(mp_image, timestamp)

            hand_detected = False

            if result.hand_landmarks:
                current_time = time.time()
                if current_time - last_time > CAPTURE_DELAY:
                    landmarks = result.hand_landmarks[0]
                    data      = normalize_landmarks(landmarks)
                    np.save(os.path.join(sign_dir, f"{start_idx + count}.npy"), data)
                    count    += 1
                    last_time = current_time
                    hand_detected = True

                # Draw skeleton
                for lm in result.hand_landmarks[0]:
                    h2, w2, _ = frame.shape
                    cx, cy = int(lm.x * w2), int(lm.y * h2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

            # HUD
            progress = count / SAMPLES
            bar_w    = 300
            bar_x, bar_y = 30, frame.shape[0] - 50
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 18), (0, 200, 120), -1)

            status_color = (0, 255, 100) if hand_detected else (0, 100, 255)
            cv2.putText(frame, f"{sign}  {count}/{SAMPLES}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
            if is_gesture:
                cv2.putText(frame, GESTURE_SIGNS[sign], (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1)
            else:
                cv2.putText(frame, "Hand detected" if hand_detected else "No hand — show your hand",
                            (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 100) if hand_detected else (0, 80, 255), 1)

            cv2.imshow("SignAI Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print(f"  ⏭  Skipped mid-collection ({count} saved)")
                break
            elif key == ord('q'):
                print(f"\n🛑 Quit. Saved {count} samples for {sign}.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        total = start_idx + count
        print(f"  ✅ Done: {sign}  ({count} new → {total} total)")

cap.release()
cv2.destroyAllWindows()

print("\n🎉 SignAI data collection complete!")
print(f"   Signs collected: {SIGNS}")
print(f"\nNext step: run  python train_model.py  to retrain.")