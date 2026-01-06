# app.py
"""
Streamlit UI
Simple web interface for the agentic RAG assistant.
"""

import streamlit as st
from agent import AgenticRAG


# Page config
st.set_page_config(
    page_title="Intent-Aware RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Intent-Aware RAG Assistant")

# Initialize agent (cached)
@st.cache_resource
def load_agent():
    """Load agent once and cache it."""
    try:
        return AgenticRAG(index_dir="./index")
    except Exception as e:
        st.error(f"Error loading agent: {e}")
        st.info("Make sure you've run ingest.py first and set GOOGLE_API_KEY environment variable.")
        return None

agent = load_agent()

# Input
user_input = st.text_input("Ask a question:", placeholder="Type your question here...")

# Button
if st.button("Ask"):
    if not user_input:
        st.warning("Please enter a question.")
    elif agent is None:
        st.error("Agent not initialized. Check setup instructions above.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = agent.query(user_input)
                
                # Display answer
                st.markdown("### Answer")
                st.write(result["final_answer"])
                
                # Display metadata in expander
                with st.expander("Show details"):
                    st.write(f"**Intent:** {result['intent']}")
                    st.write(f"**Confidence:** {result['confidence']}")
                    if result.get("original_answer"):
                        st.write(f"**Original answer:** {result['original_answer']}")
            
            except Exception as e:
                st.error(f"Error: {e}")

# Instructions
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("""
    1. Type your question in the input box
    2. Click **Ask**
    3. Wait for the response
    
    The assistant can:
    - Answer general questions
    - Search through documents
    - Perform calculations
    """)
    