import streamlit as st
from summarizer import TextSummarizer

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("📝 Text Summarization App")
st.markdown("Summarize any long piece of text using a pretrained transformer model.")

text_input = st.text_area("Enter your text here:", height=300)

if "summarizer" not in st.session_state:
    st.session_state.summarizer = TextSummarizer()

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            summary = st.session_state.summarizer.summarize(text_input)
            st.subheader("🔍 Summary:")
            st.success(summary)
