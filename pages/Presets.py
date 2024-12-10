import streamlit as st
st.header("Presets for Our Fourier Synth")
audio_val = st.audio_input("Record")
if audio_val:
    st.audio(audio_val)