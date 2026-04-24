<h1 align="center">SignAI 🤟</h1>
<h3 align="center">Real-Time Sign Language ↔ Intelligent Communication System</h3>

<p align="center">
  Developed under <b>Samsung Innovation Campus (SIC) Internship 2026</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/FastAPI-0.111-green">
  <img src="https://img.shields.io/badge/MediaPipe-0.10-orange">
  <img src="https://img.shields.io/badge/ML-SVM-yellow">

  <img src="https://img.shields.io/badge/LLM-LLaMA%203.1-purple">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## 📌 Executive Summary

**SignAI** is a real-time AI-driven communication platform that enables seamless interaction between **sign language users and spoken/written language users**.

By integrating **computer vision, machine learning, and large language models (LLMs)**, the system delivers:

* Real-time gesture recognition
* Intelligent natural language responses
* Multimodal output (text, sign animation, and speech)

This solution promotes **accessibility, inclusivity, and human-centered AI communication**.

---

## 🎯 Core Capabilities

* 🔁 **End-to-End Bi-directional Communication**
  *(Sign → Text → AI → Sign/Voice)*

* ⚡ **Low-Latency Real-Time Processing**

* 🧠 **AI-Augmented Responses**
  Powered by **LLaMA 3.1 (via Groq API)**

* 🖐 **Precision Hand Tracking**
  Using MediaPipe landmark detection

* 🔤 **Input Refinement Layer**
  Spell correction for finger-spelled text

* 🔊 **Multimodal Output System**
  Text + Sign Visualization + Speech

---

## ✨ Feature Breakdown

### 🖐 Sign Recognition Engine

* Real-time ASL/ISL detection (A–Z, 1–9)
* 83-dimensional feature extraction pipeline
* SVM-based classification model

### 💬 AI Conversation Layer

* Context-aware responses using LLM
* Natural conversational flow
* Intelligent text correction

### 🤟 Sign Visualization Engine

* Image-sequence based sign rendering
* Dynamic response-to-sign conversion

### 🎙 Speech Layer

* Integrated Text-to-Speech for accessibility

### 🖥 Interface Design

* Split-panel interactive UI:

  * Live camera input
  * Chat interface
  * Sign output display

---

## 🏗 System Architecture

```
User Gesture → MediaPipe Detection → Feature Extraction → SVM Classifier
→ Text Output → LLM Processing → Response Generation
→ Sign Animation + Speech Output
```

---

## 🛠 Technology Stack

| Layer             | Technology              |
| ----------------- | ----------------------- |
| Backend           | FastAPI, Python         |
| Machine Learning  | scikit-learn (SVM)      |
| Computer Vision   | MediaPipe               |
| LLM Integration   | Groq API (LLaMA 3.1 8B) |
| Frontend          | HTML, CSS, JavaScript   |
| Speech Processing | Text-to-Speech (TTS)    |

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/RASIQ-FARAZ-DAMBAL/SignAI.git
cd SignAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Add:

```
GROQ_API_KEY=your_api_key_here
```

### 4. (Optional) Data Collection

```bash
python collect_data.py
```

### 5. Train Model

```bash
python train_model.py
```

### 6. Run Application

```bash
uvicorn app:app --reload
```

🔗 Open: http://localhost:8000

---

## 📁 Project Structure

```
SignAI/
├── app.py
├── collect_data.py
├── train_model.py
├── export_poses.py
├── requirements.txt
├── .env.example
└── static/
    ├── index.html
    ├── sign_poses.json
    └── hands/
```

---

## 📸 Demo & Screenshots

<p align="center">
  <b>Live System Interface & Output</b>
</p>

<br>

### 🖥 Home Screen

<p align="center">
  <img alt="Screenshot 1" src="https://github.com/user-attachments/assets/15e1a4bf-74a7-4b6d-ba2c-a2a6bb4adbbe" width="100%">
</p>

<p align="center">
  <sub>Real-time interface displaying live sign detection and system layout</sub>
</p>

---

### 💬 AI Conversation Interface

<p align="center">
  <img alt="Screenshot 2" src="https://github.com/user-attachments/assets/593fe8c2-2a2a-4cb2-8390-67d4a31aa298" width="100%">
</p>

<p align="center">
  <sub>Context-aware conversational response powered by LLM integration</sub>
</p>

---

### 🤟 Sign Output Visualization

<p align="center">
  <img alt="Screenshot 3" src="https://github.com/user-attachments/assets/aa28d7b9-e5cf-433f-bcf8-6916cc86810c" width="100%">
</p>

<p align="center">
  <sub>Generated sign output representing AI responses visually</sub>
</p>

---


## 🔮 Future Roadmap

* Sentence-level sign recognition
* Deep learning models (LSTM / Transformers)
* Mobile application deployment
* Multi-language support
* Emotion-aware gesture understanding

---
## 📄 Project Documentation

<p align="center">
  <a href="[https://github.com/user-attachments/files/27043080/SignAI_Synopsis.pdf](https://github.com/RASIQ-FARAZ-DAMBAL/SignAI/blob/main/SignAI_Synopsis.pdf)">
    <img src="https://img.icons8.com/color/96/pdf.png" width="80"/>
    <br>
    <b>View Project Synopsis</b>
  </a>
</p>

---
---
## 👨‍💻 Author

**Rasiq Faraz Dambal**
Samsung Innovation Campus Intern — 2026

---

## 📜 License

This project is licensed under the MIT License.
