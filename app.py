# app.py

import streamlit as st
from old_code import GeorgianRAGSystem  # <-- adjust if your file name is different

# Must be the first Streamlit command
st.set_page_config(page_title="Georgian RAG Chatbot", page_icon="🍲")

# Initialize RAG system only once
if 'rag' not in st.session_state:
    with st.spinner("🔄 Initializing system, please wait..."):
        rag = GeorgianRAGSystem()
        rag.load_data()
        rag.setup_search()
        st.session_state.rag = rag
        st.session_state.messages = []  # Chat history
        st.success("✅ სისტემის ინიციალიზაცია დასრულდა!")

# Display title and subtitle
st.title("🍽️ Smart Bites")
st.caption("დაწერე რას შეჭამდი – მაგალითად: „შემწვარი კარტოფილი,რომელიც ბავშვობას გამახსენებს“")

# Chat container
chat_placeholder = st.container()

# Display previous messages
with chat_placeholder:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("👤 მომხმარებელი"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("🤖 ჩატბოტი"):
                st.markdown(msg["content"])

# Text input field
user_input = st.chat_input("ჩაწერე შენი შეკითხვა აქ...")

# When user sends message
if user_input:
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with chat_placeholder:
        with st.chat_message("👤 მომხმარებელი"):
            st.markdown(user_input)

    # Generate AI response
    with st.spinner("🤖 პასუხი გენერირდება..."):
        try:
            answer = st.session_state.rag.query(user_input)
        except Exception as e:
            answer = f"❌ შეცდომა: {e}"

    # Add AI response to session
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Display AI response
    with chat_placeholder:
        with st.chat_message("🤖 ჩატბოტი"):
            st.markdown(answer)
