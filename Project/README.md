# 🧠 Brain Tumor Detection System

An AI-based image classification project that detects **brain tumors** from **MRI scans** using **Convolutional Neural Networks (CNNs)** built with **TensorFlow** and deployed via **Streamlit**.

---

## 🌍 Features

- 🧠 Detects tumor vs no-tumor from MRI scans
- ⚙️ Built using TensorFlow (CNN)
- 💻 Works on Windows, macOS (Intel & M1/M2/M3), and Linux
- ☁️ Deployable to Streamlit Cloud
- 📊 95–98% accuracy on validation set

---

## 🗂️ Folder Structure
brain_tumor_app/
│
├── dataset/
│ ├── yes/
│ └── no/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md


---

## 📦 Installation & Setup

> ⚠️ Recommended: Use a **virtual environment** for isolation.

### 🪟 If you're using **Windows**
```
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow streamlit pillow numpy opencv-python
```

### If you're using macOS (Intel / M1 / M2 / M3) or Linux
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install tensorflow-macos tensorflow-metal streamlit pillow numpy opencv-python
```