import streamlit as st
import fourier_backend as fb

st.header("Preset Buttons")
st.subheader("Each button will run a different preset")
st.write("---")

audio_val = st.audio_input("Record")
if audio_val:
    meow = st.audio(audioVal)

# emale = st.checkbox(label = "Eliminate Male Voices")
# efem = st.checkbox(label = "Eliminate Female Voices")
# def presets():
#     if emale:
#         fb.elim_male()
#     if efem:
#         fb.elim_fem()
    
