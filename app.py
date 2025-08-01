import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import base64
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv
import hashlib
import pickle

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

fields = {"username": "Username", "password": "Password"}
name, auth_status, username = authenticator.login(fields=fields, location='main')

if auth_status is False:
    st.error('Username or password is incorrect')
elif auth_status is None:
    st.warning('Please enter your username and password')
elif auth_status:
    st.sidebar.success(f"Welcome, {name}!")

    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded

    def set_background(image_path):
        base64_str = get_base64_image(image_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background("background.jpg")

    st.sidebar.title("Navigator")
    page = st.sidebar.radio("Select a Page", ["Telecom Customer Churn Predictor", "Telecom Customer Churn Dashboard", "AI Assistant"])
    authenticator.logout('Logout', 'sidebar')

    if page == "Telecom Customer Churn Predictor":
        st.title('Telecom Customer Churn Predictor')

    elif page == "Telecom Customer Churn Dashboard":
        st.title('Telecom Customer Churn Dashboard')

    elif page == "AI Assistant":
        st.title('AI Assistant')

        load_dotenv()

        groq_api_key = os.getenv('GROQ_API_KEY')

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        prompt = ChatPromptTemplate.from_template(
            """
            You are a warm, friendly assistant that specializes in a certain telecom company. 
            Your goal is to be helpful while maintaining clear boundaries about your knowledge domain.
            
            Guidelines:
            1. Always be polite and approachable in your tone.
            2. If the question is answerable from the exact context:
            - Respond precisely using only the provided context
            - Keep answers clear and concise
            3. If the question is clearly related telecom (even if not in context):
            - Answer helpfully while noting when you're going beyond the provided materials
            4. For completely unrelated questions (e.g., weather, personal advice):
            - Gently explain you specialize in telecom
            - Example: "I'm afraid I don't have information about that topic."
            
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        CSV_FOLDER = "./Telecom+Customer+Churn"
        VECTORSTORE_FILE = "vectorstore.pkl"
        HASH_FILE = "content_hash.txt"

        def calculate_content_hash():
            hasher = hashlib.sha256()
            for root, _, files in os.walk(CSV_FOLDER):
                for file in files:
                    if file.endswith('.csv'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'rb') as f:
                            hasher.update(f.read())
            return hasher.hexdigest()

        def save_content_hash(content_hash):
            with open(HASH_FILE, 'w') as f:
                f.write(content_hash)

        def load_content_hash():
            if os.path.exists(HASH_FILE):
                with open(HASH_FILE, 'r') as f:
                    return f.read().strip()
            return None

        def create_vectorstore():
            embeddings = OllamaEmbeddings()
            loader = CSVLoader(CSV_FOLDER)
            docs = []

            for root, _, files in os.walk(CSV_FOLDER):
                for file in files:
                    filepath = os.path.join(root, file)
                    if os.path.isfile(filepath) and file.endswith('.csv'):
                        try:
                            loader = CSVLoader(file_path=filepath)
                            docs.extend(loader.load())
                        except Exception as e:
                            print(f"Error loading {filepath}: {e}")
                            continue

            if not docs:
                raise RuntimeError("No valid CSVs found in the directory.")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, embeddings)
            
            with open(VECTORSTORE_FILE, 'wb') as f:
                pickle.dump(vectors, f)
            
            current_hash = calculate_content_hash()
            save_content_hash(current_hash)
            
            return vectors

        def load_vectorstore():
            current_hash = calculate_content_hash()
            saved_hash = load_content_hash()
            
            if current_hash == saved_hash and os.path.exists(VECTORSTORE_FILE):
                with open(VECTORSTORE_FILE, 'rb') as f:
                    return pickle.load(f)
            
            return create_vectorstore()

        if "vectors" not in st.session_state:
            st.session_state.vectors = load_vectorstore()
        
        st.write("Ask me anything related to telecom...")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask your question..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': user_input})
            
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})