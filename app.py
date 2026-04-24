import os
import numpy as np
import joblib
import requests

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ================= LOAD .env =================
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# ================= CONFIG =================
MODEL_PATH   = "models/sign_model.pkl"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. Chat will not work.")
else:
    print(f"GROQ key loaded: {GROQ_API_KEY[:8]}...")

# ================= SPELL CHECKER =================
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    print("SpellChecker loaded.")
except ImportError:
    spell = None
    print("WARNING: pyspellchecker not installed. Run: pip install pyspellchecker")

def correct_text(text: str) -> str:
    """
    Correct each word in the text using pyspellchecker.
    Preserves original word if no correction found or spell is not available.
    """
    if not spell or not text.strip():
        return text
    words = text.split()
    corrected = []
    for word in words:
        # Only correct purely alphabetic words
        if word.isalpha():
            fix = spell.correction(word)
            corrected.append(fix if fix else word)
        else:
            corrected.append(word)
    return " ".join(corrected)

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    classes = list(model.named_steps["svm"].classes_) if hasattr(model, "named_steps") else list(model.classes_)
    print(f"Model loaded. Classes: {classes}")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


# ================= FEATURE ENGINEERING =================
# Must match train_model.py exactly — same function, same order.
FINGERTIP_IDS   = [4, 8, 12, 16, 20]
FINGER_MID_IDS  = [3, 7, 11, 15, 19]
FINGER_BASE_IDS = [2, 6, 10, 14, 18]

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def extract_features(raw):
    coords = raw.reshape(21, 3)
    features = list(raw)
    for tip, mid, base in zip(FINGERTIP_IDS, FINGER_MID_IDS, FINGER_BASE_IDS):
        features.append(angle_between(coords[tip], coords[mid], coords[base]) / 180.0)
    for i in range(len(FINGERTIP_IDS)):
        for j in range(i + 1, len(FINGERTIP_IDS)):
            features.append(np.linalg.norm(coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]]))
    for tip in FINGERTIP_IDS:
        features.append(np.linalg.norm(coords[tip] - coords[0]))
    return np.array(features)  # 83 features


# ================= PREDICT =================
@app.post("/predict")
async def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        raw      = np.array(data["features"])
        features = extract_features(raw).reshape(1, -1)
        pred     = model.predict(features)[0]
        prob     = float(model.predict_proba(features).max())
        return {"prediction": pred, "confidence": prob}
    except Exception as e:
        return {"error": str(e)}



# ================= CHAT =================
GROQ_MODELS = [
    "llama-3.1-8b-instant",              # fastest — try first always
    "llama-3.3-70b-versatile",           # fallback
    "meta-llama/llama-4-scout-17b-16e-instruct",  # last resort
]

# In-memory chat history
chat_history = []
MAX_HISTORY  = 10

@app.post("/chat")
async def chat(data: dict):
    global chat_history
    try:
        raw_text  = data.get("message", "").strip()
        from_sign = data.get("from_sign", False)

        if not raw_text:
            return {"reply": "No message received."}
        if not GROQ_API_KEY:
            return {"reply": "Error: GROQ_API_KEY not set in .env file"}

        # Inline spell correct — no extra round trip to /correct
        user_text = correct_text(raw_text) if from_sign else raw_text
        corrected = (user_text != raw_text)

        chat_history.append({"role": "user", "content": user_text})
        if len(chat_history) > MAX_HISTORY * 2:
            chat_history = chat_history[-(MAX_HISTORY * 2):]

        url     = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }

        system_msg = {
            "role": "system",
            "content": (
                "You are SignAI, an AI assistant for sign language users. "
                "Keep responses short and clear — 1-3 sentences max unless detail is needed. "
                "Never mention sign language unless the user asks."
            )
        }

        last_error = "Unknown error"

        for model_name in GROQ_MODELS:
            try:
                payload = {
                    "model":      model_name,
                    "messages":   [system_msg] + chat_history,
                    "max_tokens": 200,      # reduced from 300 — faster response
                    "temperature": 0.7,
                    "stream":     False
                }
                response = requests.post(url, headers=headers, json=payload, timeout=10)  # 10s not 30s
                result   = response.json()

                if "choices" in result:
                    reply = result["choices"][0]["message"]["content"]
                    chat_history.append({"role": "assistant", "content": reply})
                    print(f"✓ {model_name}")
                    return {
                        "reply":     reply,
                        "corrected": corrected,
                        "used_text": user_text
                    }
                else:
                    last_error = result.get("error", {}).get("message", "No choices")
                    print(f"✗ {model_name}: {last_error}")

            except requests.Timeout:
                print(f"✗ {model_name}: timeout")
                continue
            except Exception as e:
                print(f"✗ {model_name}: {e}")
                continue

        return {"reply": f"Connection issue. Please try again."}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


@app.post("/clear_history")
async def clear_history():
    """Clear conversation history — called on new chat."""
    global chat_history
    chat_history = []
    return {"status": "cleared"}


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)