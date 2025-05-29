# Chatyon
Developed an interactive chat application with PDF upload support, enabling users to extract text from  multiple PDFs, generate embeddings, and store them in MongoDB for efficient retrieval and analysis. The application  leverages Azure OpenAI for conversational AI and Sentence Transformers for embedding generation.

import streamlit as st
from langchain_community.chat_models import AzureChatOpenAI
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # For generating embeddings
from pymongo import MongoClient


# Azure OpenAI Configuration
azure_deployment_name = "gpt-4o"
azure_openai_api_key = "9nh861Kj4U5vBzSqHCfbwJ1EloO8rhXOKoVt9WE8qqIH0Wcd5YwZJQQJ99ALACHYHv6XJ3w3AAAAACOGubrO"
azure_openai_base_url = "https://stackyon-ai-services.openai.azure.com/"

# MongoDB Configuration
mongo_connection_string = "mongodb://localhost:27017"  # Replace with your MongoDB connection string
mongo_database_name = "sai1db"  # Database name
mongo_collection_name = "embeddings"  # Collection name
try:
    # Ensure there are no whitespaces in the connection string
    mongo_connection_string = mongo_connection_string.strip()
    client = MongoClient(mongo_connection_string)
    db = client[mongo_database_name]
    collection = db[mongo_collection_name]
    print("Connected successfully to MongoDB!")
except ValueError as e:
    print(f"MongoDB connection string error: {e}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Initialize Azure Chat Model
chat_model = AzureChatOpenAI(
    deployment_name=azure_deployment_name,
    azure_endpoint=f"{azure_openai_base_url}",
    openai_api_key=azure_openai_api_key,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
    temperature=0.7,
    max_tokens=512,
)

# Set up Streamlit page configuration
st.set_page_config(page_title="Chat App with PDF Support", layout="wide")

# Persistent session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""
if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = None

st.title("CHATYON")

# PDF File Upload Section (Multiple Files)
uploaded_files = st.file_uploader(
    "Upload PDF files for context (optional)",
    type=["pdf"],
    accept_multiple_files=True  # Allow multiple file uploads
)

if uploaded_files:
    text_content = ""

    # Extract text from all uploaded PDFs
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text_content += page.extract_text()

    # Store the combined PDF content in session state
    st.session_state.pdf_content = text_content
    st.success(f"Loaded {len(uploaded_files)} PDF(s) successfully!")

    # Generate embeddings for the PDF content
    if st.button("Convert PDF to Embeddings"):
        with st.spinner("Generating embeddings..."):
            # Initialize the SentenceTransformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Split the text into chunks (optional, but recommended for large PDFs)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = text_splitter.split_text(st.session_state.pdf_content)

            # Generate embeddings for each chunk
            embeddings = model.encode(text_chunks)

            # Store embeddings in session state
            st.session_state.pdf_embeddings = embeddings
            st.success("PDF embeddings generated successfully!")

            # Store embeddings in MongoDB
            if st.session_state.pdf_embeddings is not None:
                # Create a document to insert into MongoDB
                embedding_document = {
                    "pdf_content": st.session_state.pdf_content,  # Store the original text
                    "embeddings": st.session_state.pdf_embeddings.tolist(),  # Convert numpy array to list
                    "num_chunks": len(text_chunks),  # Number of text chunks
                }
                # Insert the document into MongoDB
                collection.insert_one(embedding_document)
                st.success("Embeddings stored in MongoDB successfully!")

# Display the conversation in ChatGPT-like format
for message in st.session_state.messages:
    if message["sender"] == "user":
        st.markdown(f'<div style="text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;">{message["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: left; background-color: #ECECEC; padding: 10px; border-radius: 10px; margin: 5px;">{message["text"]}</div>', unsafe_allow_html=True)

# Add a bottom-aligned input box with a form
with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...", placeholder="Ask a question based on PDF or general queries...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Append user message
    st.session_state.messages.append({"sender": "user", "text": user_input})

    try:
        # Build context if PDF content is available
        context_prompt = st.session_state.pdf_content + "\n" + user_input if st.session_state.pdf_content else user_input

        # Get bot response
        response = chat_model.predict(context_prompt)

        # Append bot message
        st.session_state.messages.append({"sender": "bot", "text": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

    # Scroll to the bottom for new messages
    st.rerun()

# Display embeddings if they exist
if st.session_state.pdf_embeddings is not None:
    st.subheader("PDF Embeddings")
    st.write(st.session_state.pdf_embeddings)

# Fetch and display embeddings from MongoDB
if st.button("Show Embeddings from MongoDB"):
    # Fetch all documents from the MongoDB collection
    embeddings_from_db = collection.find()

    # Display each document
    for doc in embeddings_from_db:
        st.subheader("Embedding Document from MongoDB")
        st.write(f"Number of Chunks: {doc['num_chunks']}")
        st.write(f"PDF Content: {doc['pdf_content'][:500]}...")  # Display first 500 characters of the text
        st.write(f"Embeddings: {doc['embeddings'][:2]}...")  # Display first 2 embeddings for brevity
