import streamlit as st
from st_custom_components import st_audiorec
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import wave
import io
import os
import matplotlib.pyplot as plt
import pylab

# loading the saved model 
new_model = tf.keras.models.load_model('saved_model/my_model')
# new_model.summary()

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# For every recording, make a spectogram and save it as label_speaker_no.png
# if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
#     os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))

# Talking to the model
def woodcutting_sound(wav_audio_data):
    sound_info, frame_rate = get_wav_info(wav_audio_data)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(f'{wav_audio_data}.png')
    fp = f'{wav_audio_data}.png'
    pylab.close()
    img = image.load_img(fp,target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = new_model.predict(img_preprocessed)
    print(prediction)
    x_labels = ['0', '1']
    plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    plt.show()

def audiorec_demo_app():
    # TITLE
    st.title('Audio Classifier')

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data

    wav_audio_data = st_audiorec()

    # add some spacing and informative messages
    col_info, col_space = st.columns([0.57, 0.43])
    with col_info:
        st.write('\n')  # add vertical spacer
        st.write('\n')  # add vertical spacer
        st.write('The processed .wav audio data, as received in the backend Python code, is displayed below.')

    if wav_audio_data is not None:
        # display audio data as received on the Python side
        col_playback, col_space = st.columns([0.58,0.42])
        with col_playback:
            st.audio(wav_audio_data, format='audio/wav')
        
        if st.button("Predict"): 
            output = woodcutting_sound(wav_audio_data)
            st.success(f"Predicted Output: {output}")


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()
    # Add a file uploader for audio files
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

# Function to read audio file and return data
def read_audio_file(audio_file):
    with io.BytesIO() as buffer:
        buffer.write(audio_file.read())
        buffer.seek(0)
        data, samplerate = sf.read(buffer, dtype='float32')
    return data, samplerate


# If an audio file has been uploaded, display its information and play it
if audio_file is not None:
    # Read the audio file data and sample rate
    data, samplerate = read_audio_file(audio_file)

    # Display some information about the audio file
    st.write("Audio file information:")
    st.write(f" - File name: {audio_file.name}")
    st.write(f" - File type: {audio_file.type}")
    st.write(f" - Sample rate: {samplerate}")
    st.write(f" - Duration: {len(data)/samplerate:.2f} seconds")

    if st.button("Predict"): 
        output = woodcutting_sound(audio_file)
        st.success(f"Predicted Output: {output}")

    # Add an audio player to play the uploaded file
    try:
        st.audio(data, format=audio_file.type, sample_rate=samplerate)
    except:
        print("Error")    
