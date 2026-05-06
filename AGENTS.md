\# Sentinel AI Agent Instructions



\## Project Overview

Sentinel AI is a Flask/PyTorch retinal image analysis service used by Sentinel Clinic.



\## Main File

\- `app/app.py` contains the Flask API and `/analyze` endpoint.



\## Current Responsibilities

\- Validate fundus images.

\- Run fundus gate model.

\- Run diabetic retinopathy classifier.

\- Return referral risk, confidence, severity, heatmap and clinician-support notes.



\## Clinical Safety Rules

\- Never present AI output as a final diagnosis.

\- Prefer clinician review over false reassurance.

\- Do not claim maculopathy is assessed unless a dedicated maculopathy model exists.

\- Keep disclaimers in responses.

\- Do not remove fundus gate checks.

\- Preserve heatmap and processed image outputs.

\- Sentinel AI may say `Clinician Review Recommended` for borderline cases.

\- Current model is not a dedicated maculopathy detector.



\## Current Limitations

\- The model is not lesion-segmentation based.

\- The model is not a dedicated maculopathy classifier.

\- The model uses 224x224 inference unless retrained.

\- Small lesions may be missed.



\## Commands

\- Run locally: `python app/app.py`

\- Commit changes from repo root.



\## Coding Rules

\- Keep API response backwards compatible.

\- Add new fields rather than removing existing fields.

\- Keep JSON serializable values.

\- Avoid hardcoded local Windows paths in production code.

