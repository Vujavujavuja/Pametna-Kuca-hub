# 🗣️ Deep CNN Audio Classification: Making Sound Understandable

## Project Name

**Deep CNN Audio Classifier** (or, if you want to be fancy: **MFCC/Mel-Powered Speech Recognizer**)

***

## Project Overview

This is a typical pipeline for deep learning-based **audio classification** or **speech recognition**. We grab audio files, turn the messy sound waves into cleaner numerical features, and then throw those features into a deep **Convolutional Neural Network (CNN)** for training.

The goal is to automatically classify speech or audio segments based on what's spoken—we don't write down the words, we just tag the audio. It's built with TensorFlow/Keras, so it's ready to chew through data fast, especially with GPU support.

***

## ⚙️ How It Works (The Core Steps)

It's a two-stage process. We didn't reinvent the wheel, we just made a reliable path from `.wav` file to trained model.

### 1. Feature Extraction (The Hard Part)

We use the standard audio analysis library **Librosa** to pull out two key features from raw audio:

* **MFCC (Mel-Frequency Cepstral Coefficients):** Extracts the mean and standard deviation of 13 coefficients. These are the classic, compact features for speech.
* **Mel-Spectrogram:** A visual representation of sound frequency over time, converted to a log scale. We flatten this big array and combine it with the MFCCs.

The features are then padded or truncated to a fixed `max_length` and pickled to a file (`labeled_features.pkl`) so we don't have to wait for Librosa again.

### 2. Deep CNN Model Training (The Deep Part)

We use **TensorFlow/Keras** to build and train a multi-layered CNN.

* **Architecture:** It's an "advanced deep CNN" with multiple 2D convolutional layers, **Batch Normalization**, **LeakyReLU** activations, and heavy **Dropout** for regularization. Crucially, it uses a **Residual Connection** to help the network train deeper without vanishing gradients.
* **Optimization:** Uses the **Adam** optimizer, with **Early Stopping** and a **ReduceLROnPlateau** callback to manage training efficiently.
* **Data Prep:** Audio features are reshaped into 2D "images" (with depth 1) so the CNN can process them like any other image-based task.

***

## 🛠️ Technology Stack (Must-Haves)

| Component | Purpose |
| :--- | :--- |
| **Python** | Everything. |
| **Librosa** | Audio loading and feature extraction. |
| **TensorFlow / Keras** | Building and training the Deep CNN model. |
| **NumPy / Pandas** | Data wrangling and array manipulation. |
| **Scikit-learn** | Data splitting and label encoding. |

***

## 🚀 Getting Started

Just make sure you have your dependencies and your data in the right spot.

### Data Structure

This project expects your audio files and transcripts to be organized like this:
├── transcript.txt # Contains 'filename.wav\tlabel' lines 
├── extracted_audio/ 
│ └── proj_dataset/ # Directory containing all your .wav files 
├── labeled_features.pkl # Output from the feature extraction step 
├── main_script.py # This is the main code 
└── README.md


### Installation

```bash
# Good luck with this part
pip install librosa numpy pandas scikit-learn tensorflow
```
How to Run
Extract Features: Run the script once. It will load, process, and save all features to labeled_features.pkl.

Train Model: Run the script again. It loads the pickle file, trains the deep CNN, and saves the final model and label encoder.

Bash
```
python main_script.py
```
