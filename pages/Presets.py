import streamlit as st

from fourier_backend import fourier

st.header("Presets for Our Fourier Synth")

incon = st.container(border=True)
outcon = st.container(border=True)

# record audio
audio_val = incon.audio_input("Record")
if audio_val:
    incon.audio(audio_val)

# presets
preset_dict = {}
emale = incon.checkbox("Eliminate Male Voices")
if emale:
    preset_dict["emale"] = True
else:
    preset_dict["emale"] = False
efemale = incon.checkbox("Eliminate Female Voices")
if efemale:
    preset_dict["efemale"] = True
else:
    preset_dict["efemale"] = False
ebird = incon.checkbox("Eliminate Bird Noises")
if ebird:
    preset_dict["ebird"] = True
else:
    preset_dict["ebird"] = False
# @TODO

# process audio
incon.button("Process", type="primary", on_click=fourier, kwargs={"audio_obj": audio_val, "presets": preset_dict, "outcon": outcon})

