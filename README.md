# Sound Classification

Sound Classification is a Python application that allows you to collect audio samples, label them, and train a deep learning model for sound classification. It provides a graphical user interface (GUI) for easy interaction and data collection.

## Features

- Collect audio samples using the microphone and loopback virtual audio device.
- Label the collected audio samples with specific categories.
- Train a deep learning model for sound classification.
- Real-time feedback on the number of collected samples.
- Save and load trained models.

## Requirements

- Python 3.6+
- Libraries: tensorflow, keras, soundfile, sounddevice, librosa, tkinter

## Installation

1. Clone the repository:

2. Install the required libraries:


## Usage

1. Launch the application:


2. Use the GUI to perform the desired actions:

- Enter a label for the audio samples you want to collect.
- Click the "Train Model" button to train the sound classification model.
- Click the "Detect and Label Sound" button to start collecting audio samples and label them in real-time.

## File Structure

- `main.py`: Main script to launch the application.
- `sound_model.h5`: Pre-trained sound classification model (generated after training).
- `labeled_audio/`: Directory to store the labeled audio samples.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


