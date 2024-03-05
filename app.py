import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from tqdm import tqdm
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, render_template, request

# Import Whisper model related utilities
import torch
import torchaudio
from typing import List, Union, Callable

# Define paths and constants
SAMPLING_RATE = 16_000
win_length = 400  # int((25 / 1_000) * SAMPLING_RATE)
hop_length = 160  # int((10 / 1_000) * SAMPLING_RATE)

device = "cuda" if torch.cuda.is_available() else "cpu"

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=128,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=128,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=80,
    n_stft=257,
    sample_rate=SAMPLING_RATE,
).to(device)

delta_fn = torchaudio.transforms.ComputeDeltas(
    win_length=400,
    mode="replicate",
)

class Trainer:
    # Define Trainer class
    pass

def get_frontend(frontends: List[str]) -> Union[torchaudio.transforms.MFCC, torchaudio.transforms.LFCC, Callable,]:
    if "mfcc" in frontends:
        return prepare_mfcc_double_delta
    elif "lfcc" in frontends:
        return prepare_lfcc_double_delta
    raise ValueError(f"{frontends} frontend is not supported!")

def prepare_lfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = LFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
    return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)

def prepare_mfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = MFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
    return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)

# Flask app initialization
app = Flask(__name__)

# Define frontend and trainer objects
frontend = get_frontend(["mfcc"])
trainer = Trainer()

# Define the features_extractor function
def features_extractor(file):
    audio, sample_rate = torchaudio.load(file)
    audio = audio.squeeze()
    features = frontend(audio)
    return features

# Define the rec route
@app.route('/rec', methods=["POST", "GET"])
def rec():
    # Load the Whisper model weights
    whisper_model = load_whisper_model()

    if request.method == "POST":
        audio = request.files['file']
        features = features_extractor(audio)
        
        # Make prediction using the Whisper model
        with torch.no_grad():
            prediction = whisper_model(features)
            prediction = torch.sigmoid(prediction).item()

        # Determine if voice is AI-generated or not based on prediction threshold
        confidence_threshold = 0.5  # Adjust as needed
        is_ai_generated = prediction > confidence_threshold

        msg = "AI-generated" if is_ai_generated else "Not AI-generated"
            
        return render_template('rec.html', msg=msg, confidence=prediction)
    
    return render_template('rec.html')

# Function to load Whisper model
def load_whisper_model():
    model = None  # Load your Whisper model here
    return model

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
