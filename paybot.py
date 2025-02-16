import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import random

def initialize_api_keys():
    if 'GOOGLE_API_KEY' not in st.session_state:
        st.session_state['GOOGLE_API_KEY'] = ''
    
    with st.sidebar:
        st.title("API Key Configuration")
        api_key = st.text_input(
            "Enter your api key",
            value=st.session_state['GOOGLE_API_KEY'],
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if api_key != st.session_state['GOOGLE_API_KEY']:
            st.session_state['GOOGLE_API_KEY'] = api_key
            if 'conversation' in st.session_state:
                del st.session_state['conversation']

def setup_llm():
    with st.sidebar:
        llm_choice = st.radio(
            "Choose LLM:",
            ["Gemini", "LLAMA-2", "Mistral-7B"]
        )
    
    if llm_choice == "Gemini":
        if not st.session_state['GOOGLE_API_KEY']:
            st.sidebar.error("Please enter your Google API key in the sidebar")
            st.stop()
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=st.session_state['GOOGLE_API_KEY'],
            temperature=0.7
        )
    else:
        if not os.getenv('HUGGINGFACEHUB_API_TOKEN'):
            st.sidebar.error("Please set HUGGINGFACEHUB_API_TOKEN environment variable")
            st.stop()
        
        model_map = {
            "LLAMA-2": "meta-llama/Llama-2-7b-chat-hf",
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        
        return HuggingFaceHub(
            repo_id=model_map[llm_choice],
            model_kwargs={"temperature": 0.7, "max_length": 512},
            huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
        )

def create_chat_container():
    # Create a container for the chat messages with custom styling
    chat_container = st.container()
    with chat_container:
        st.markdown("""
            <style>
                .chat-container {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    height: 400px;
                    overflow-y: auto;
                    background-color: #f9f9f9;
                }
                .message {
                    margin-bottom: 10px;
                    padding: 10px;
                    border-radius: 5px;
                }
                .user-message {
                    background-color: #007AFF;
                    color: white;
                    margin-left: 20%;
                    margin-right: 5px;
                }
                .bot-message {
                    background-color: #E5E5EA;
                    color: black;
                    margin-right: 20%;
                    margin-left: 5px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        if 'messages' in st.session_state:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="message user-message">{message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="message bot-message">{message["content"]}</div>', 
                              unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation' not in st.session_state:
        llm = setup_llm()
        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            prompt=PAYMENT_EXPERT_PROMPT,
            memory=ConversationBufferMemory()
        )

PAYMENT_EXPERT_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are VK18, an expert in payments technology. You have deep knowledge about payment systems, 
    digital payments, payment processing, financial technology, and related areas.

    If the query is not related to payments, respond with a joke. If it is related to payments, provide a detailed, accurate response.

    Conversation History:
    {history}

    Human: {input}
    VK18: """
)

def get_response(user_input):
    try:
        return st.session_state['conversation'].predict(input=user_input)
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def main():
    st.set_page_config(
        page_title="VK18 - Payment Expert",
        page_icon="💳",
        layout="wide"
    )
    
    initialize_api_keys()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("VK18 - Your Payment Technology Expert")
        st.markdown("""
        👋 Hi! I'm VK18, your payment technology expert. I can help you with:
        - Payment processing systems
        - Digital payment solutions
        - Payment security
        - Transaction flows
        - Payment infrastructure
        
        For non-payment queries, I'll share a fun joke with you! 😊
        """)
        
        initialize_session_state()
        
        # Create chat container
        create_chat_container()
        
        # Create a form for user input
        with st.form(key='message_form', clear_on_submit=True):
            user_input = st.text_area("Type your message:", key='user_input', height=100)
            submit_button = st.form_submit_button("Send Message")
            
            if submit_button and user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get bot response
                response = get_response(user_input)
                
                # Add bot response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update chat
                st.rerun()

if __name__ == "__main__":
    main()