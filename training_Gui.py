
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import datetime
import soundfile as sf
import os
import tkinter as tk
from tkinter import messagebox
import threading
import time
import sounddevice as sd
import librosa


# Set up the audio streams for microphone and loopback virtual audio device
chunk_size = 16000  # Number of samples in each audio chunk
audio_format = 'flac'
channels = 1
rate = 16000
desired_duration_per_sample = chunk_size / rate
total_samples = 10000  # Total number of samples to collect

# Create a directory to store the labeled audio files
label_dir = 'labeled_audio'
os.makedirs(label_dir, exist_ok=True)

# Create a label dictionary
label_dict = {
    'konshens': 0,
    'Ariana': 1,
    'drake': 2,
    'shensea': 3,
    # Add more labels and their corresponding numerical values as needed
}

# Generate a dataset from the labeled audio files
def generate_dataset():
    X = []
    y = []
    for label in label_dict:
        label_dir_path = os.path.join(label_dir, label)
        if os.path.exists(label_dir_path):
            files = os.listdir(label_dir_path)
            for file in files:
                audio_path = os.path.join(label_dir_path, file)
                try:
                    audio_data, sr = librosa.load(audio_path, sr=rate, mono=True)
                except Exception as e:
                    print(f"Error loading audio file: {audio_path}")
                    continue
                if len(audio_data) == chunk_size:
                    audio_data = audio_data.reshape((chunk_size, 1))
                    X.append(audio_data)
                    y.append(label_dict[label])
                else:
                    print(f"Invalid audio length for file: {audio_path}")
    if len(X) == 0 or len(y) == 0:  # Check if either X or y is empty
        print("No data available for training.")
        return None, None
    X = np.array(X, dtype=np.float32)
    X /= np.max(np.abs(X))
    y = tf.keras.utils.to_categorical(y, num_classes=len(label_dict))
    return X, y


# Check if the model file exists
model_file = 'sound_model.h5'
if os.path.isfile(model_file):
    model = load_model(model_file)
else:
    messagebox.showinfo("Info", "Model file not found. Starting training...")

# Define the model architecture
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(chunk_size, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(label_dict), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to preprocess audio data for training
def preprocess_audio_data(audio_data):
    audio_data /= np.max(np.abs(audio_data))
    audio_data = np.expand_dims(audio_data, axis=0)
    audio_data = np.expand_dims(audio_data, axis=2)
    return audio_data


# Train the model
def train_model():
    # Generate the dataset
    X_train, y_train = generate_dataset()

    if X_train is None or y_train is None:
        messagebox.showinfo("Info", "No data available for training.")
        return

    if len(X_train) == 0:
        messagebox.showinfo("Info", "No data available for training.")
        return

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[ModelCheckpoint('sound_model.h5', save_best_only=True)])

    messagebox.showinfo("Info", "Training complete.")

# Collect audio data in a separate thread
def collect_audio_data(label, label_dir_path):
    count = 0
    stop_collection = False

    def audio_processing(indata, frames, time, status):
        nonlocal count
        nonlocal stop_collection
        if stop_collection or count >= total_samples:
            return
        audio_frames = np.asarray(indata[:, 0], dtype=np.float32)
        audio_frames /= np.max(np.abs(audio_frames))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(label_dir_path, f"{label}_{timestamp}_{count}.{audio_format}")
        sf.write(file_path, audio_frames, rate, format=audio_format)
        count += 1
        count_label.config(text=f"Data collected: {count}/{total_samples}")
        count_label.update()

    def start_collection():
        nonlocal stop_collection
        stop_button.config(state="normal")
        stop_collection = False

        # Start the loopback audio stream
        loopback_stream = sd.InputStream(callback=audio_processing, channels=channels, samplerate=rate, dtype='float32')
        loopback_stream.start()

        # Start the microphone audio stream
        mic_stream = sd.InputStream(callback=audio_processing, channels=channels, samplerate=rate, dtype='float32')
        mic_stream.start()

        return loopback_stream, mic_stream

    def stop_collection_func():
        nonlocal stop_collection
        stop_button.config(state="disabled")
        stop_collection = True

    def collect_data():
        loopback_stream, mic_stream = start_collection()
        start_time = time.time()
        while not stop_collection and count < total_samples:
            if time.time() - start_time > desired_duration_per_sample:
                break
            time.sleep(0.01)  # Wait for a small interval to avoid freezing
        loopback_stream.stop()
        mic_stream.stop()
        stop_collection_func()

    stop_button = tk.Button(window, text="Stop Collection", command=stop_collection_func, state="disabled")
    stop_button.grid(row=3, column=1, pady=10)

    collect_data_thread = threading.Thread(target=collect_data)
    collect_data_thread.start()

# Detect and label sound category
def detect_and_label_sound():
    label = label_entry.get()
    if not label:
        messagebox.showinfo("Error", "Please enter a label.")
        return

    label_dir_path = os.path.join(label_dir, label)
    os.makedirs(label_dir_path, exist_ok=True)

    count = 0
    stop_collection = False

    collect_audio_data(label, label_dir_path)

# GUI
window = tk.Tk()
window.title("Sound Classification")
window.geometry("700x700")

# Add the label entry field
label_entry = tk.Entry(window)
label_entry.grid(row=0, column=0, padx=10, pady=10)

# Add the train button
train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.grid(row=1, column=0, padx=10, pady=10)

# Add the detect and label button
detect_button = tk.Button(window, text="Detect and LabelSound", command=detect_and_label_sound)
detect_button.grid(row=2, column=0, padx=10, pady=10)

# Add the count label
count_label = tk.Label(window, text="Data collected: 0/10000")
count_label.grid(row=3, column=0, padx=10, pady=10)

window.mainloop()
