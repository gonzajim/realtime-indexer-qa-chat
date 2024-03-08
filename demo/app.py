import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from endpoint_utils import get_inputs


DRIVE_URL = os.environ.get(
    "GDRIVE_FOLDER_URL",
    "https://drive.google.com/drive/folders/1FXLRopdqEn3VxZyEgghsxhsN7_ZrZ9UQ",
)



# Load environment variables
load_dotenv()


# Streamlit UI elements
st.write(
    "## UCLM ReCaVa Chatbot ⚡ RAG en S3"
)


htt = """
<p>
    <span> Arquitectura: </span>
    <img src="./app/static/combinedhosted.png" width="300" alt="Google Drive Logo">
</p>
"""
st.markdown(htt, unsafe_allow_html=True)


image_width = 300
image_height = 200


if "messages" not in st.session_state.keys():
    from llama_index.llms.types import ChatMessage, MessageRole
    from rag import chat_engine, vector_client

    welcome_message = "Escribe tu mensaje"
    DEFAULT_MESSAGES = [
        ChatMessage(role=MessageRole.USER, content="Bienvenido al chatbot conversacional del observatorio de la UCLM sobre diligencia debida en Sostenibilidad."),
        ChatMessage(role=MessageRole.ASSISTANT, content=welcome_message),
    ]
    chat_engine.chat_history.clear()

    for msg in DEFAULT_MESSAGES:
        chat_engine.chat_history.append(msg)

    st.session_state.messages = [
        {"role": msg.role, "content": msg.content} for msg in chat_engine.chat_history
    ]
    st.session_state.chat_engine = chat_engine
    st.session_state.vector_client = vector_client


results = get_inputs()

last_modified_time, last_indexed_files = results


df = pd.DataFrame(last_indexed_files, columns=[last_modified_time, "status"])

df.set_index(df.columns[0])
st.dataframe(df, hide_index=True, height=150, use_container_width=True)

cs = st.columns([1, 1, 1, 1], gap="large")

with cs[-1]:
    st.button("⟳ Refresh", use_container_width=True)


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            sources = []

            try:
                for source in response.source_nodes:
                    full_path = source.metadata.get("path", source.metadata.get("name"))
                    if full_path is None:
                        continue
                    if "/" in full_path:
                        name = f"`{full_path.split('/')[-1]}`"
                    else:
                        name = f"`{full_path}`"
                    if name not in sources:
                        sources.append(name)
            except AttributeError:
                print(f"No source (`source_nodes`) was found in response: {response}")

            sources_text = ", ".join(sources)

            response_text = (
                response.response
                + f"\n\nDocuments looked up to obtain this answer: {sources_text}"
            )

            st.write(response_text)

            message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(message)
