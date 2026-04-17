# SkinAura AI

## Modular Computer Vision System for Dermatological Analysis and Recommendation

---

## Overview

SkinAura AI is an end-to-end computer vision system for analyzing facial skin conditions and generating structured skincare recommendations.

The system integrates a **transfer learning–based CNN (MobileNetV2)** with a **FastAPI inference service** and a **Streamlit analytical dashboard**, with emphasis on:

* Interpretable outputs through confidence distributions
* Converting model predictions into actionable context
* Clear separation between model, API, and UI layers

---

## Problem Framing

Most machine learning projects stop at classification.

SkinAura extends this by:

* translating probabilistic outputs into **severity levels**
* exposing **confidence-aware predictions**
* producing **structured routines instead of raw labels**

This shifts the system from a model demo to a **decision-support pipeline**.

---

## System Architecture

```text
Streamlit Dashboard (UI Layer)
        ↓
FastAPI Backend (Inference & Orchestration)
        ↓
CNN Model (MobileNetV2)
        ↓
Post-processing & Recommendation Engine
```

---

## Core Components

**Frontend (Streamlit)**
Handles image input, visualization, and structured result rendering.

**Backend (FastAPI)**
Performs request validation, preprocessing, model inference, and response construction.

**Model Layer**
MobileNetV2-based classifier fine-tuned for dermatological conditions.

**Recommendation Engine**
Rule-based logic mapping predictions and confidence scores to step-based routines.

---

## Key Technical Features

### Interpretable Predictions

* Primary prediction with confidence
* Secondary class signal
* Confidence distribution across all classes

Designed to avoid opaque outputs and support interpretation.

---

### Severity Assessment

```text
Severity = f(confidence, thresholds)
```

Maps model confidence into:

* Low
* Moderate
* High

---

### Modular Pipeline

The workflow is decomposed into:

* Image preprocessing
* Model inference
* Prediction ranking
* Recommendation generation

Each stage is independently replaceable.

---

### Structured Outputs

The system returns:

* Condition classification
* Confidence score
* Severity level
* Step-based skincare routine (AM / PM)

---

## Tech Stack

| Layer            | Technologies                   |
| ---------------- | ------------------------------ |
| Machine Learning | TensorFlow, Keras, MobileNetV2 |
| Backend          | FastAPI, Uvicorn, Pydantic     |
| Frontend         | Streamlit (custom CSS)         |
| Image Processing | OpenCV, PIL                    |
| Data Handling    | NumPy                          |

---

## Model Specifications

* Base Model: MobileNetV2 (ImageNet pretrained)
* Input Size: 224 × 224 × 3

**Classification Head:**

* Global Average Pooling
* Dense Layer (ReLU)
* Dropout (regularization)
* Softmax output

**Classes:**

* Acne
* Acne Scars
* Pigmentation
* Normal

---

## Running the System

### Backend

```bash
uvicorn backend.main:app --reload
```

### Frontend

```bash
streamlit run app_dashboard.py
```

---

## Deployment

The system supports a decoupled deployment setup:

* Backend: FastAPI (e.g., Render)
* Frontend: Streamlit (Streamlit Cloud)

The frontend communicates with the backend via REST APIs.

---

## Limitations

* Dataset size and diversity may affect generalization
* No clinical validation
* Recommendation engine is rule-based

---

## Future Work

* Face region detection and localized analysis
* Improved dataset diversity
* Learned recommendation system
* Model optimization for lower latency

---

## Disclaimer

This system is intended for educational and decision-support purposes only.
It does not provide medical diagnoses.

---

## Author

Sujanya Srinivas
GitHub: https://github.com/sujanya-hub

---

## Summary

This project demonstrates:

* End-to-end ML system design
* Model + API + UI integration
* Interpretable output design
* Practical, product-oriented AI development

---
