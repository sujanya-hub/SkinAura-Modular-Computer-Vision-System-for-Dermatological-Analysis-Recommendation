# SkinAura AI

## Modular Computer Vision System for Dermatological Analysis and Personalized Skincare Recommendations

## Overview

SkinAura AI is an end-to-end computer vision system designed to analyze facial skin conditions and generate personalized skincare recommendations in real time.

The project combines deep learning, image preprocessing, backend APIs, and an interactive frontend into a complete deployment pipeline. The goal was not just to build a classification model, but to create a practical AI system that converts predictions into structured, user-friendly skincare insights.

The system uses a CNN-based classifier trained on curated dermatological image datasets and integrates preprocessing techniques to improve prediction reliability across different lighting conditions and image qualities.

---

# Problem Framing

Most skin analysis ML projects stop at predicting labels.

SkinAura extends this by:

* exposing confidence-aware predictions
* converting predictions into severity levels
* generating structured AM/PM skincare routines
* mapping conditions to relevant skincare ingredients
* separating model inference from frontend presentation

This shifts the project from a simple classification demo to a more complete decision-support system.

---

# System Architecture

```text id="k5stlc"
Streamlit Dashboard (Frontend/UI)
        ↓
FastAPI Backend (Inference API)
        ↓
CNN Classification Model
        ↓
Post-processing & Recommendation Engine
```

---

# Core Components

## Frontend — Streamlit Dashboard

Handles:

* image uploads
* prediction visualization
* confidence display
* skincare recommendation rendering
* user interaction workflow

---

## Backend — FastAPI

Responsible for:

* request validation
* image preprocessing
* model inference
* response generation
* API orchestration

---

## Model Layer

CNN-based classifier built using TensorFlow/Keras for dermatological condition classification.

### Detection Categories

* Acne
* Acne Scars
* Pigmentation
* Normal Skin
* Texture Irregularities

### Model Performance

| Metric              | Value                 |
| ------------------- | --------------------- |
| Validation Accuracy | 95–96%                |
| Dataset Size        | 3,500+ Curated Images |
| Model Type          | CNN                   |
| Inference Mode      | Real-Time             |

---

## Recommendation Engine

Rule-based recommendation system that maps predictions and confidence levels into skincare guidance.

### Output Includes

* condition classification
* confidence score
* severity level
* AM/PM skincare routine
* ingredient suggestions

---

# Image Processing Pipeline

To improve model consistency across different image conditions, the system includes a preprocessing pipeline using OpenCV.

### Techniques Used

* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Adaptive lighting normalization
* Image resizing and normalization
* Tensor preprocessing

These steps help improve texture visibility and reduce lighting-related prediction instability.

---

# Key Technical Features

## Interpretable Predictions

The system provides:

* primary prediction
* confidence score
* secondary prediction signals
* confidence distribution across classes

This was designed to avoid completely opaque model outputs and make predictions easier to interpret.

---

## Severity Assessment

The recommendation engine maps prediction confidence into severity levels:

* Low
* Moderate
* High

This adds additional context beyond simple classification labels.

---

## Modular Pipeline Design

The workflow is separated into independent stages:

1. Image preprocessing
2. Model inference
3. Prediction ranking
4. Recommendation generation
5. Frontend visualization

Each stage can be modified or replaced independently.

---

# Tech Stack

| Layer            | Technologies               |
| ---------------- | -------------------------- |
| Machine Learning | TensorFlow, Keras          |
| Computer Vision  | OpenCV, PIL                |
| Backend          | FastAPI, Uvicorn, Pydantic |
| Frontend         | Streamlit                  |
| Data Processing  | NumPy, Pandas              |
| Deployment       | Render, Streamlit Cloud    |
| Version Control  | Git, GitHub                |

---

# Model Specifications

| Component         | Details                    |
| ----------------- | -------------------------- |
| Base Architecture | CNN                        |
| Framework         | TensorFlow/Keras           |
| Input Size        | 224 × 224 × 3              |
| Output Layer      | Softmax Classification     |
| Inference Type    | Multi-Class Classification |

---

# Running the Project

## Backend

```bash id="51o77g"
uvicorn backend.main:app --reload
```

---

## Frontend

```bash id="r1e24f"
streamlit run app_dashboard.py
```

---

# Deployment

| Service           | Link                                                                                                                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frontend          | [https://skinaura-ai.streamlit.app/](https://skinaura-ai.streamlit.app/)                                                                                                                                                               |
| Backend API       | [https://skinaura-backend.onrender.com](https://skinaura-backend.onrender.com)                                                                                                                                                         |
| GitHub Repository | [https://github.com/sujanya-hub/SkinAura-Modular-Computer-Vision-System-for-Dermatological-Analysis-Recommendation](https://github.com/sujanya-hub/SkinAura-Modular-Computer-Vision-System-for-Dermatological-Analysis-Recommendation) |

The frontend and backend are deployed separately to support cleaner scaling and modular service management.

---

# Current Limitations

* Dataset diversity can still be improved
* No clinical validation
* Recommendation system is currently rule-based
* Performance may vary depending on image quality

---

# Future Work

Planned improvements include:

* Vision Transformer (ViT) integration
* U-Net segmentation for localized analysis
* Real-time webcam inference
* Mobile application support
* Learned recommendation systems
* Improved dataset diversity
* Lower-latency inference optimization

---

# Disclaimer

This project is intended for educational and AI research purposes only. It is not a medical diagnostic tool and should not replace professional dermatological advice.

---

# Author

**Sujanya Srinivas**
GitHub: [https://github.com/sujanya-hub](https://github.com/sujanya-hub)

---

# Project Summary

This project demonstrates:

* end-to-end ML system design
* frontend + backend AI integration
* real-time inference deployment
* interpretable prediction systems
* practical computer vision workflows
* product-oriented AI engineering
