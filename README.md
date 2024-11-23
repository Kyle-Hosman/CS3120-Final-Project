# Music Genre Classification Web App

## Overview
This project leverages a **Random Forests** model to classify music genres. 

## Dataset
The app utilizes the **GTZAN Dataset** from Kaggle, which includes:
- **10 Genres** with 100 audio files per genre (30 seconds each).
- Features extracted into two CSV files:
  - **30-second files:** Contains mean and variance of audio features.
  - **3-second files:** Splits songs into smaller segments, providing 10x the data.

## Functionality
The web app allows users to:
1. **Upload an audio clip**.
2. Automatically **extract relevant audio features** from the clip.
3. Pass these features to the **Random Forest model** for genre prediction.
4. Display the **predicted genre** to the user.

## Key Features
- **Fast and efficient predictions** using Random Forests.
- Support for user-uploaded audio clips.
- **Real-time feature extraction** for accurate predictions.
