# CropCast AI - Project Summary
## Problem
Farmers need an accessible way to estimate crop yield and identify disease-risk conditions from field measurements.

## Inputs
Rainfall, temperature, humidity, soil moisture, soil pH, NPK, irrigation, fertilizer, pest level, disease severity and NDVI.

## Outputs
1. Estimated crop yield in tons/hectare.
2. Disease risk: Low, Medium or High.
3. Recommendation for field monitoring.

## ML pipeline
Data Loading -> EDA -> Cleaning -> IQR Outlier Treatment -> Feature Engineering -> Encoding -> Scaling -> Train/Test Split -> Model Training -> Evaluation -> Prediction.

## Academic note
Model performance on the included synthetic data must not be interpreted as agricultural field accuracy. For a real deployment, train and validate on region-specific agronomic data.