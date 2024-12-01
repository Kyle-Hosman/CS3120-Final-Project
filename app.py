from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pandas as pd
import pickle


# Initialize the Flask app
app = Flask(__name__)

genre_mapping = {
    0: "classical",
    1: "jazz",
    2: "country",
    3: "pop",
    4: "rock",
    5: "hiphop",
    6: "metal",
    7: "blues",
    8: "reggae",
    9: "disco",
}

# Load the trained model
model_filename = "random_forest_model.pkl"
with open(model_filename, "rb") as file:
    model, feature_names = pickle.load(file)


# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    # Length of the audio signal
    length = len(y)

    # Extract spectral and temporal features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))

    # Add chroma_stft features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # Extract harmony and percussive features
    harmony = np.mean(librosa.effects.harmonic(y))
    harmony_var = np.var(librosa.effects.harmonic(y))
    perceptr = np.mean(librosa.effects.percussive(y))
    perceptr_var = np.var(librosa.effects.percussive(y))

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)  # Compute mean for each MFCC
    mfccs_var = np.var(mfccs, axis=1)  # Compute variance for each MFCC

    # Extract RMS
    rms = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))

    # Extract tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Combine features
    features = np.hstack(
        [
            length,
            chroma_stft_mean,
            chroma_stft_var,
            rms,
            rms_var,
            spectral_centroid,
            spectral_centroid_var,
            spectral_bandwidth,
            spectral_bandwidth_var,
            rolloff,
            rolloff_var,
            zero_crossing_rate,
            zero_crossing_rate_var,
            harmony,
            harmony_var,
            perceptr,
            perceptr_var,
            tempo,
            mfccs_mean,
            mfccs_var,
        ]
    )

    # Check feature count
    print("Extracted features count:", len(features))
    return features


# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")


# Route for handling file uploads and making predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    file_path = "temp_audio.wav"
    file.save(file_path)

    # Extract features and make prediction
    features = extract_features(file_path)
    features_df = pd.DataFrame([features], columns=feature_names)  # Use feature names
    prediction = model.predict(features_df)
    # features = features.reshape(1, -1)  # Reshape for the model
    # prediction = model.predict(features)

    # Convert prediction to native Python type and map to genre name
    genre_label = int(prediction[0])
    genre_name = genre_mapping.get(genre_label, "Unknown")

    # Return the prediction result
    return jsonify({"genre": genre_name})


# Run the app
if __name__ == "__main__":
    app.run()