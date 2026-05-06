# SkinAura AI
### Modular Computer Vision System for Dermatological Analysis & Skincare Recommendations

<p align="center">
  Real-time skin condition analysis, confidence-aware predictions, and personalized skincare recommendations.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-CNN-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://skinaura-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-success?style=for-the-badge" />
  </a>

  <a href="https://skinaura-backend.onrender.com">
    <img src="https://img.shields.io/badge/Backend-Render-purple?style=for-the-badge" />
  </a>
</p>

---

## Live Deployment

| Service | Link |
|---|---|
| Frontend | https://skinaura-ai.streamlit.app/ |
| Backend API | https://skinaura-backend.onrender.com |
| GitHub Repository | https://github.com/sujanya-hub/SkinAura-Modular-Computer-Vision-System-for-Dermatological-Analysis-Recommendation |

---

## Overview

SkinAura AI is an end-to-end computer vision system built for facial skin condition analysis and personalized skincare recommendation generation.

The platform combines:
- image preprocessing
- CNN-based classification
- confidence-aware predictions
- severity estimation
- recommendation generation
- frontend + backend deployment

into a modular real-time inference pipeline.

The project focuses on converting raw model predictions into structured and interpretable skincare guidance instead of limiting the workflow to label classification alone.

---

## Preview

### Upload & Image Analysis Interface

Image upload workflow with preprocessing, real-time inference, and prediction handling.

![Upload Interface](assets/upload-analysis.png)

---

### Skin Profile Configuration

User profile selection and contextual skincare configuration workflow.

![Skin Profile](assets/skin-profile.png)

---

### AI Analysis Dashboard

Confidence-aware predictions, severity assessment, preprocessing visualization, and personalized skincare insights.

![Analysis Dashboard](assets/analysis-dashboard.png)

---

### Recommendation Engine

Structured skincare recommendations including routines, ingredient suggestions, and severity-aware guidance.

![Recommendation Engine](assets/recommendation-engine.png)

---

## Problem Framing

Most dermatology-related ML projects stop at classification outputs.

SkinAura extends this workflow by:
- exposing confidence-aware predictions
- generating severity-based interpretations
- creating structured skincare routines
- mapping skin conditions to ingredients
- separating inference from recommendation logic
- building a deployable frontend + backend pipeline

This shifts the project from a model demo into a practical AI-assisted recommendation system.

---

## System Architecture

```text
          ┌─────────────────────────┐
          │ Streamlit Frontend UI  │
          └──────────┬─────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │     FastAPI Backend     │
          │   Inference Endpoints   │
          └──────────┬─────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │ CNN Classification Model│
          └──────────┬─────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │ Recommendation Engine   │
          │ + Severity Mapping      │
          └─────────────────────────┘
```

---

## Core Components

### Frontend — Streamlit Dashboard

Handles:
- image uploads
- prediction visualization
- confidence rendering
- skincare recommendation display
- user interaction workflows

### Backend — FastAPI

Responsible for:
- request validation
- image preprocessing
- model inference
- API orchestration
- response formatting

### Model Layer

CNN-based classifier built using TensorFlow/Keras for dermatological condition classification.

### Recommendation Engine

Rule-based recommendation pipeline that converts predictions into:
- skincare guidance
- severity levels
- ingredient recommendations
- AM/PM skincare routines

---

## Detection Categories

- Acne
- Acne Scars
- Pigmentation
- Normal Skin
- Texture Irregularities

---

## Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | 95–96% |
| Dataset Size | 3,500+ Curated Images |
| Model Type | CNN |
| Inference Mode | Real-Time |

---

## Image Processing Pipeline

To improve inference consistency across different image conditions, the system includes preprocessing using OpenCV.

### Techniques Used

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive lighting normalization
- Image resizing and normalization
- Tensor preprocessing

These steps help improve texture visibility and reduce lighting-related prediction instability.

---

## Interpretable Prediction Design

Instead of returning only a single label, the system exposes:
- primary prediction
- confidence score
- secondary prediction probabilities
- confidence distributions
- severity mapping

This improves interpretability and makes outputs easier to evaluate.

---

## Engineering Decisions

| Decision | Reasoning |
|---|---|
| CNN-based architecture | Lightweight and efficient for real-time inference |
| OpenCV preprocessing pipeline | Improves robustness across varying lighting conditions |
| Frontend/backend separation | Cleaner deployment and modular scaling |
| Confidence-aware outputs | Improves interpretability over single-label prediction |
| Rule-based recommendation layer | Keeps recommendation generation explainable |
| FastAPI inference endpoints | Simplifies API orchestration and deployment |

---

## Technical Stack

| Layer | Technologies |
|---|---|
| Machine Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, PIL |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Data Processing | NumPy, Pandas |
| Deployment | Render, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Model Specifications

| Component | Details |
|---|---|
| Base Architecture | CNN |
| Framework | TensorFlow/Keras |
| Input Size | 224 × 224 × 3 |
| Output Layer | Softmax Classification |
| Inference Type | Multi-Class Classification |

---

## Project Structure

```text
SkinAura/
│
├── assets/
│   ├── upload-analysis.png
│   ├── skin-profile.png
│   ├── analysis-dashboard.png
│   └── recommendation-engine.png
│
├── backend/
├── frontend/
├── models/
├── utils/
├── uploads/
│
├── app_dashboard.py
├── requirements.txt
└── README.md
```

---

## Running the Project

### Backend

```bash
uvicorn backend.main:app --reload
```

---

### Frontend

```bash
streamlit run app_dashboard.py
```

---

## Current Limitations

- Dataset diversity can still be improved.
- No clinical or dermatological validation.
- Recommendation engine is currently rule-based.
- Performance depends heavily on image quality and lighting conditions.
- The system is not optimized for extremely low-light or blurry images.

---

## Planned Improvements

- Vision Transformer (ViT) integration
- U-Net segmentation for localized analysis
- Real-time webcam inference
- Mobile application support
- Learned recommendation systems
- Improved dataset diversity
- Lower-latency inference optimization

---

## Example Use Cases

### Personalized Skincare Assistance
Generate skincare recommendations based on detected conditions.

### Educational Computer Vision Demo
Demonstrate end-to-end image classification pipelines.

### AI-Powered Skin Analysis
Provide confidence-aware dermatological condition predictions.

### ML Deployment Demonstration
Showcase modular frontend/backend AI deployment workflows.

---

## Disclaimer

This project is intended for educational and AI research purposes only.

It is not a medical diagnostic system and should not replace professional dermatological advice.

---

## Developer

### Sujanya Srinivas

AI/ML Engineer focused on:
- Computer Vision Systems
- AI Deployment Pipelines
- Real-Time Inference Systems
- Applied Deep Learning
- Full-Stack AI Applications

---

## License

MIT License
