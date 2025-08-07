import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    # Page configuration
    st.set_page_config(
        page_title="MediBot - AI Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stChatInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #667eea;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏥AI Medical Chatbot </h1>
        <p>Your AI-Powered Medical Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📋 About MediBot")
        st.markdown("""
        <div class="sidebar-info">
        <p><strong>🎯 Purpose:</strong> Get reliable medical information from trusted sources</p>
        <p><strong>📚 Knowledge Base:</strong> Encyclopedia of Medicine</p>
        <p><strong>🔒 Privacy:</strong> Your conversations are secure</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Be specific about your symptoms
        - Ask about medical conditions, treatments, or procedures
        - Use clear, simple language
        - Ask one question at a time
        """)

        st.markdown("### ⚠️ Important Notice")
        st.warning("This AI assistant provides general medical information only. Always consult with healthcare professionals for medical advice, diagnosis, or treatment.")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        👋 **Welcome to MediBot!**

        I'm here to help you with medical information based on trusted medical references.

        **What can I help you with today?**
        - Symptoms and conditions
        - Medical procedures
        - Treatment information
        - General health questions

        Feel free to ask me anything medical-related! 🩺
        """
        st.session_state.messages.append({'role': 'assistant', 'content': welcome_msg})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input with custom placeholder
    prompt = st.chat_input("💬 Ask me about any medical condition, symptom, or treatment...")

    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        # Show typing indicator
        with st.chat_message('assistant'):
            with st.spinner('🔍 Searching medical database...'):
                pass

        CUSTOM_PROMPT_TEMPLATE = """
                You are a medical AI assistant. Use the provided medical context to answer the user's question accurately and concisely.

                Guidelines:
                - Provide clear, direct answers based only on the given context
                - If you don't know the answer, say "I don't have enough information to answer that question"
                - Use simple, easy-to-understand language
                - Focus on the most relevant information
                - Do not include technical document references or metadata in your response

                Context: {context}
                Question: {question}

                Answer:
                """
         

        #TODO: Create a Groq API key and add it to .env file
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]

            # Create a clean, formatted response
            clean_result = f"🩺 **Medical Information:**\n\n{result}"

            # Add source information in an attractive format
            if source_documents:
                clean_result += f"\n\n---\n📚 **Source:** Based on {len(source_documents)} medical reference(s) from Encyclopedia of Medicine"

            # Display the response with better formatting
            with st.chat_message('assistant'):
                st.markdown(clean_result)

                # Add a small success indicator
                st.success("✅ Information retrieved from medical database")

            st.session_state.messages.append({'role':'assistant', 'content': clean_result})

        except Exception as e:
            with st.chat_message('assistant'):
                st.error("🚨 **Oops! Something went wrong**")
                st.markdown(f"**Error Details:** {str(e)}")
                st.info("💡 **Suggestion:** Try rephrasing your question or check your internet connection.")

            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({'role':'assistant', 'content': error_msg})

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏥 <strong>MediBot</strong> - Powered by AI & Medical Knowledge Base</p>
        <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()