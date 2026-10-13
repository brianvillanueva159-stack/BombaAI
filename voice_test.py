import streamlit as st
from streamlit_mic_recorder import speech_to_text

st.title("🎤 BombaAI Voice Test")

voice_text = speech_to_text(
    language="en",
    start_prompt="🎤 Start Voice",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    key="voice_test"
)

if voice_text:
    st.success("Voice recognized!")
    st.write("You said:")
    st.write(voice_text)