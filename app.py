from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from gtts.lang import tts_langs
import streamlit as st
import os
from dotenv import load_dotenv  # For environment variable management

# Load environment variables
load_dotenv()

# Retrieve the Google API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found! Make sure you have a .env file with GOOGLE_API_KEY.")

# Default language settings
if "language" not in st.session_state:
    st.session_state.language = "en"
if "prompt_language" not in st.session_state:
    st.session_state.prompt_language = "English"

# Set page configuration
st.set_page_config(page_title="AI Voice Assistant", page_icon="🤖")

# App title and description
st.title("Bilingual AI Voice Assistant 🎙️")
st.subheader("Interact in Urdu/English with Real-Time Voice Input")

# Instructions
st.write("Please select a language mode and start speaking:")

# Adjust column widths for closer buttons
col1, col2 = st.columns([1, 1])  # Use equal proportions to minimize distance

with col1:
    if st.button("Urdu"):
        st.session_state.language = "ur"
        st.session_state.prompt_language = "Urdu"
        print("Urdu mode selected")

with col2:
    if st.button("English"):
        st.session_state.language = "en"
        st.session_state.prompt_language = "English"
        print("English mode selected")

# Chat prompt template
print(st.session_state.prompt_language)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            f"You are a helpful AI assistant. Please always respond to user queries in pure "
            + st.session_state.prompt_language
            + " language."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Initialize chat message history
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Load the AI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the chain
chain = prompt | model | StrOutputParser()

# Add history handling
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Process voice input
with st.spinner("Converting Speech To Text..."):
    text = speech_to_text(language=st.session_state.language, use_container_width=True, just_once=True, key="STT")

if text:
    st.chat_message("human").write(text)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Pass `prompt_language` and `question` dynamically
        response = chain_with_history.stream(
            {"question": text, "prompt_language": st.session_state.prompt_language}, {"configurable": {"session_id": "any"}}
        )

        for res in response:
            full_response += res or ""
            message_placeholder.markdown(full_response)

    with st.spinner("Converting Text To Speech..."):
        tts = gTTS(text=full_response, lang=st.session_state.language)
        tts.save("output.mp3")
        st.audio("output.mp3")
else:
    st.warning("Please press the button and start speaking.")
