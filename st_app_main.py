import streamlit as st
from st_custom_components import st_audiorec
import soundfile as sf
import numpy as np
import pickle
import io

# loading the saved model 
# loaded_model = pickle.load(open('C:/Users/KIIT/Desktop/Internship/streamlit/streamlit_audio_recorder/trained_model.sav', 'rb'))

# Talking to the model
def woodcutting_sound(wav_audio_data, audio_file):
    pass

# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
st.set_page_config(page_title="Audio Classifier")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -5rem; }</style>''',
            unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # lightmode

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

    # Add an audio player to play the uploaded file
    try:
        st.audio(data, format=audio_file.type, sample_rate=samplerate)
    except:
        print("Error")    
