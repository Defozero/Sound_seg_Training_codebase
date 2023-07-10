import numpy as np
from keras.models import load_model
import os
import pyaudio
import tkinter as tk

# Set up the audio stream from the inbuilt microphone
chunk_size = 8000
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
stream = pyaudio.PyAudio().open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

# Load the trained model
model = load_model('sound_model.h5')

# Create a label dictionary (should match the labels used during training)
label_dict = {
    0: 'test2',
    1: 'test1',
    2: 'dancehall',
    3: 'house',
    # Add more labels as needed
}

# Preprocess audio data for prediction
def preprocess_audio_data(audio_data):
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    audio_data = np.expand_dims(audio_data, axis=0)
    audio_data = np.expand_dims(audio_data, axis=2)
    return audio_data

# Detect and label sound category
def detect_and_label_sound():
    audio_frames = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
    audio_data = preprocess_audio_data(audio_frames)
    predicted_label = np.argmax(model.predict(audio_data), axis=1)
    if predicted_label[0] in label_dict:
        predicted_category = label_dict[predicted_label[0]]
    else:
        predicted_category = "Foreign Sound"
    sound_label.config(text=f"Detected sound category: {predicted_category}")
    window.after(1000, detect_and_label_sound)  # Perform detection every 1 second

# Create a GUI window
window = tk.Tk()
window.title("Sound Detection")
window.geometry("400x200")

# Create a label to display the detected sound category
sound_label = tk.Label(window, text="Detected sound category:")
sound_label.pack()

# Start continuous sound detection
detect_and_label_sound()

# Run the GUI event loop
window.mainloop()
