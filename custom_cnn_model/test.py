import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
import pickle
import tempfile

SAMPLE_RATE = 16000
DURATION = 3
TEMP_WAV_PATH = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

MODEL_PATH = "advanced_deep_cnn_attention_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def extract_features(file_path, max_length=5000, feature_height=50):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_features = np.hstack((mfcc_mean, mfcc_std))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mel_flattened = log_mel.flatten()

    # Combine features
    combined_features = np.hstack((mfcc_features, mel_flattened))

    if len(combined_features) > max_length:
        combined_features = combined_features[:max_length]
    elif len(combined_features) < max_length:
        combined_features = np.pad(combined_features, (0, max_length - len(combined_features)), mode='constant')

    feature_width = max_length // feature_height
    reshaped_features = combined_features.reshape((1, feature_height, feature_width, 1))
    return reshaped_features


def record_audio():
    print("Recording... Speak into your microphone.")
    recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    write(TEMP_WAV_PATH, SAMPLE_RATE, recording)
    print(f"Audio recorded and saved to {TEMP_WAV_PATH}")


def predict_command(file_path):
    features = extract_features(file_path)
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)
    command = label_encoder.inverse_transform([predicted_label])[0]
    confidence = predictions[0][predicted_label]
    return command, confidence


if __name__ == "__main__":
    record_audio()

    command, confidence = predict_command(TEMP_WAV_PATH)
    print(f"Predicted Command: {command} (Confidence: {confidence:.2f})")

    os.remove(TEMP_WAV_PATH)
