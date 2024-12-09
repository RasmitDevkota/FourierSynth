import streamlit as st
import fourierBackend as fb

st.header("Preset Buttons")
st.subheader("Each button will run a different preset")
st.write("---")

audioVal = st.audio_input("Record")
if audioVal:
    st.audio(audioVal)

emale = st.checkbox(label = "Eliminate Male Voices")
efem = st.checkbox(label = "Eliminate Female Voices")
def presets():
    if emale:
        fb.elimMale()
    if efem:
        fb.elimFemale()
    
