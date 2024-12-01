# Music Genre Classification Web App

<div align="center">
  <img src="screenshot.png" alt="Application Screenshot" width="400">
</div>

## Overview
This project leverages a **Random Forests** model to classify music genres. 

## Dataset
The app utilizes the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) from Kaggle, which includes:
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

# How to Use this App
1. Create a Python Virtual Environment:
`python -m venv .venv`
2. Activate the Virtual Environment:
`.venv\Scripts\activate`
3. Install Dependencies:
`pip install -r requirements.txt`
4. Run the web app with:
`python app.py`
5. Open http://127.0.0.1:5000 in your browser to see the app
6. Take a look at the [**Notebook.ipynb**](https://github.com/Kyle-Hosman/CS3120-Final-Project/blob/main/Notebook.ipynb) to see how it all works!

## Limitations
- The model relies on pre-extracted features, which limits how well it can understand the full complexity of audio data. The features I used may not capture the subtle differences between genres very effectively.
- When I tried adding new features, like rhythm or tone-based ones, these didn’t improve the model much, showing that just using features like this with Random Forest has its limits.
- If I had more time, using a CNN could greatly improve accuracy because it can better understand the detailed patterns in audio.
