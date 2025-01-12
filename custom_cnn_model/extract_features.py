import librosa
import numpy as np
import os
import pandas as pd


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.hstack((mfcc_mean, mfcc_std))


def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel


def feature_extraction(file_path):
    mfcc = extract_mfcc(file_path)
    mel = extract_mel(file_path)

    mel_flattened = mel.flatten()

    combined_features = np.hstack((mfcc, mel_flattened))

    return combined_features


def process_dataset(txt_file_path, audio_dir, output_file):
    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_labels = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            file_labels[parts[0]] = parts[1]

    data = []
    for file_name, label in file_labels.items():
        file_path = os.path.join(audio_dir, file_name)
        if os.path.exists(file_path):
            print(f"Processing {file_name}...")
            try:
                features = feature_extraction(file_path)
                data.append((features, label))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_name}")

    df = pd.DataFrame(data, columns=["features", "label"])
    df.to_pickle(output_file)
    print(f"Feature extraction complete. Data saved to {output_file}.")


if __name__ == "__main__":
    txt_file_path = "transcript.txt"
    audio_dir = "extracted_audio/proj_dataset"
    output_file = "labeled_features.pkl"

    process_dataset(txt_file_path, audio_dir, output_file)
