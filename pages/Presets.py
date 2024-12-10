import streamlit as st

from src/fourier_backend import fourier

st.header("Presets for Our Fourier Synth")
audio_val = st.audio_input("Record")
if audio_val:
    st.audio(audio_val)


st.button("Process", type="primary", on_click=fourier)

