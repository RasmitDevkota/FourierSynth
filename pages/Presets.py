import streamlit as st

from fourier_backend import fourier

st.header("Presets for Our Fourier Synth")

# record audio

audio_val = st.audio_input("Record")
if audio_val:
    st.audio(audio_val)

# presets
preset_list = []
emale = st.checkbox("Eliminate Male Voices")
if emale:
    preset_list.append()
# @TODO

# process audio

st.button("Process", type="primary", on_click=fourier, kwargs={"audio_obj": audio_val, "presets": preset_list})

