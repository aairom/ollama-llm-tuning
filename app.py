import streamlit as st
import ollama
import requests
import json
import subprocess
import os
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Docling components are removed as per user request

# Load environment variables from .env file
load_dotenv()

# Check if environment variables are loaded
if not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID") or not os.environ.get("IBM_CLOUD_API_URL"):
    st.warning(
        "**Environment Variables Not Loaded!** "
        "Please ensure your `.env` file is in the same directory as `app.py` "
        "and contains `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and `IBM_CLOUD_API_URL`."
    )

# --- Configuration Constants ---
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "ollama_ibm_rag_docs"
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Ollama model for embeddings
EMBEDDING_DIM = 768 # Dimension for nomic-embed-text embeddings

# --- Known IBM Cloud Watsonx.ai Regions (for dynamic LLM lookup) ---
IBM_WATSONX_AI_REGIONS = {
    "us-south": "Dallas",
    "eu-de": "Frankfurt",
    "jp-tok": "Tokyo",
    "eu-gb": "London",
    "au-syd": "Sydney",
    "ca-tor": "Toronto",
    "ap-south-1": "Mumbai"
}

# --- Helper Functions ---
@st.cache_resource
def get_ollama_client():
    """Returns an Ollama client instance."""
    return ollama

def get_local_ollama_models():
    """Fetches a list of locally available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models_data = response.json()
        return [model["name"] for model in models_data.get("models", [])]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Please ensure Ollama is running and accessible on http://localhost:11434.")
        return []
    except Exception as e:
        st.error(f"Error fetching Ollama models: {e}")
        return []

def get_ibm_watsonx_ai_llms(region_code, auth_token, project_id):
    """
    Fetches a list of foundation models available in a specific IBM Watsonx.ai region.
    Requires a valid IAM token.
    """
    api_url = f"https://{region_code}.ml.cloud.ibm.com/ml/v1/foundation_model_specs"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    params = {
        "version": "2024-05-01"
    }

    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        models_data = response.json()
        
        llm_ids = []
        if "resources" in models_data:
            for resource in models_data["resources"]:
                if "model_id" in resource:
                    llm_ids.append(resource["model_id"])
        return llm_ids
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 401:
            st.error("IBM Cloud API Authentication Error: Invalid or expired access token. Please update your 'Authorization Header'.")
        elif http_err.response.status_code == 403:
            st.error("IBM Cloud API Permission Error: Your token does not have sufficient permissions to access this resource.")
        else:
            st.error(f"IBM Cloud API HTTP Error: {http_err}. Response: {http_err.response.text}")
        return []
    except requests.exceptions.ConnectionError:
        st.error(f"IBM Cloud API Connection Error: Could not connect to {api_url}. Check your network and URL.")
        return []
    except json.JSONDecodeError:
        st.error("IBM Cloud API Error: Failed to decode JSON response. The API might have returned an unexpected format.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching IBM Cloud LLMs: {e}")
        return []

def test_milvus_connection():
    """Tests the connection to Milvus and returns a status message."""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias="test_conn") # Use a temporary alias
        if utility.has_collection(MILVUS_COLLECTION_NAME, using="test_conn"):
            status = f"Milvus connection successful! Collection '{MILVUS_COLLECTION_NAME}' exists."
        else:
            status = f"Milvus connection successful! Collection '{MILVUS_COLLECTION_NAME}' does NOT exist (will be created on first document upload)."
        connections.disconnect("test_conn") # Disconnect the temporary alias
        return status, True
    except Exception as e:
        return f"Milvus connection failed: {e}", False

def get_milvus_collection_and_connect():
    """Establishes connection and returns the Milvus collection, creating it if necessary."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT) # Establish the main connection
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        collection = Collection(MILVUS_COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535) # Max length for VARCHAR
        ]
        schema = CollectionSchema(fields, MILVUS_COLLECTION_NAME)
        collection = Collection(MILVUS_COLLECTION_NAME, schema)

        # Create an index for the vector field for efficient similarity search
        index_params = {
            "metric_type": "COSINE", # Cosine similarity for embeddings
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
    return collection

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Splits text into chunks with overlap."""
    chunks = []
    if not text:
        return chunks
    
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_text_ollama(text_chunks):
    """Generates embeddings for text chunks using Ollama's nomic-embed-text."""
    embeddings = []
    ollama_client = get_ollama_client()
    try:
        # Check if the embedding model is available locally without re-pulling if exists
        local_ollama_models = get_local_ollama_models()
        if EMBEDDING_MODEL_NAME not in local_ollama_models:
            st.sidebar.warning(f"Embedding model '{EMBEDDING_MODEL_NAME}' not found locally. Attempting to pull...")
            pull_command = f"ollama pull {EMBEDDING_MODEL_NAME}"
            process = subprocess.Popen(
                pull_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                st.sidebar.text(line.strip()) # Display pull progress in sidebar
            process.wait()
            if process.returncode != 0:
                st.error(f"Failed to pull embedding model '{EMBEDDING_MODEL_NAME}'. RAG will not work.")
                return []
            else:
                st.sidebar.success(f"Successfully pulled '{EMBEDDING_MODEL_NAME}'.")
        # No message if model is already found, to reduce clutter.

        for chunk in text_chunks:
            response = ollama_client.embeddings(model=EMBEDDING_MODEL_NAME, prompt=chunk)
            embeddings.append(response['embedding'])
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings with Ollama: {e}")
        return []

def insert_documents_into_milvus(documents):
    """Inserts documents (text chunks and their embeddings) into Milvus."""
    try:
        collection = get_milvus_collection_and_connect() # Get collection and ensure connection is active
        
        # Prepare data for insertion
        data = [
            documents["embeddings"],
            documents["texts"]
        ]
        
        # Insert data
        mr = collection.insert(data)
        collection.flush() # Ensure data is written to disk
        connections.disconnect("default") # Disconnect after operation
        return True, f"Successfully inserted {len(documents['texts'])} chunks into Milvus."
    except Exception as e:
        # Ensure disconnection even on error
        try:
            connections.disconnect("default")
        except Exception:
            pass # Ignore if disconnect fails
        return False, f"Error inserting documents into Milvus: {e}"

def search_milvus(query_embedding, top_k=3):
    """Searches Milvus for relevant document chunks."""
    try:
        collection = get_milvus_collection_and_connect() # Get collection and ensure connection is active
        
        # Explicitly check if collection exists before proceeding (redundant with get_milvus_collection_and_connect but good for clarity)
        if not utility.has_collection(MILVUS_COLLECTION_NAME):
            connections.disconnect("default")
            raise Exception(f"Milvus collection '{MILVUS_COLLECTION_NAME}' does not exist. Please process documents first.")

        collection.load() # Load collection into memory for search

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        retrieved_texts = [hit.entity.get("text") for hit in results[0]]
        collection.release() # Release collection from memory
        connections.disconnect("default") # Disconnect after operation
        return retrieved_texts
    except Exception as e:
        # Ensure disconnection even on error
        try:
            connections.disconnect("default")
        except Exception:
            pass # Ignore if disconnect fails
        # Re-raise the exception after logging for broader error handling
        raise e

# --- Streamlit GUI ---
st.set_page_config(layout="wide", page_title="Local/IBM Cloud LLM Interface")
st.title("LLM Interface: Local Ollama & IBM Cloud")

# --- Add your logo here ---
st.image(
    "./images/wxai-localstudio.png",
    caption="LLM Studio",
    width=200
)
st.markdown("---") # Separator after logo

# --- Sidebar for Configuration ---
st.sidebar.header("LLM Target Configuration")
target = st.sidebar.radio(
    "Choose LLM Target:",
    ("Local Ollama", "IBM Cloud")
)

# --- Conditional fields based on target ---
selected_model_id = None
ibm_cloud_url = ""
ibm_cloud_project_id = ""
ibm_cloud_authorization = ""
ibm_cloud_llm_model_name = ""
ibm_cloud_version = ""

if target == "IBM Cloud":
    st.sidebar.subheader("IBM Cloud Settings")
    
    # Get default URL from environment or fallback
    env_ibm_cloud_url = os.environ.get("IBM_CLOUD_API_URL")
    if env_ibm_cloud_url:
        initial_ibm_cloud_url = env_ibm_cloud_url
        url_help_text = "The API endpoint for IBM Cloud LLM inference. Loaded from IBM_CLOUD_API_URL env var."
    else:
        initial_ibm_cloud_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text"
        url_help_text = "The API endpoint for IBM Cloud LLM inference. Set via IBM_CLOUD_API_URL env var or manually."

    ibm_cloud_url = st.sidebar.text_input(
        "IBM Cloud API Endpoint URL (for inference)",
        value=initial_ibm_cloud_url,
        help=url_help_text
    )

    # Get default Project ID from environment or fallback
    env_project_id = os.environ.get("WATSONX_PROJECT_ID")
    if env_project_id:
        initial_project_id = env_project_id
        project_id_help_text = "Your IBM Cloud Project ID. Loaded from WATSONX_PROJECT_ID env var."
    else:
        initial_project_id = "YOUR_IBM_CLOUD_PROJECT_ID"
        project_id_help_text = "Replace with your actual IBM Cloud Project ID. Can be set via WATSONX_PROJECT_ID env var."

    ibm_cloud_project_id = st.sidebar.text_input(
        "IBM Cloud Project ID",
        value=initial_project_id,
        help=project_id_help_text
    )

    # Check for WATSONX_API_KEY environment variable
    default_api_key = os.environ.get("WATSONX_API_KEY")
    if default_api_key:
        # Do not add "Bearer" here, it will be added in the API call
        default_auth_header_value = default_api_key
    else:
        default_auth_header_value = "YOUR_ACCESS_TOKEN"

    ibm_cloud_authorization = st.sidebar.text_input(
        "Authorization Header (IAM Token)",
        value=default_auth_header_value,
        help="Replace 'YOUR_ACCESS_TOKEN' with a valid IBM Cloud IAM token. Can be set via WATSONX_API_KEY env var. Do NOT include 'Bearer' prefix here."
    )
    
    ibm_cloud_llm_model_name = st.sidebar.text_input(
        "IBM Cloud LLM Model Name (for inference)",
        value="meta-llama/llama-3-3-70b-instruct", # Updated default model name
        help="Specify the model ID available on your IBM Cloud instance (e.g., 'google/flan-ul2', 'meta-llama/llama-3-3-70b-instruct')."
    )

    # New field for IBM Cloud API Version
    ibm_cloud_version = st.sidebar.text_input(
        "IBM Cloud API Version (e.g., 2023-05-29)",
        value="", # Default to blank as requested
        help="Specify the API version for IBM Cloud LLM inference calls. Format: YYYY-MM-DD. This is added to the payload if not empty."
    )

else: # Local Ollama
    st.sidebar.subheader("Local Ollama Settings")
    all_local_ollama_models = get_local_ollama_models()
    
    # Filter out the embedding model from the chat model selection
    chat_ollama_models = [model for model in all_local_ollama_models if model != EMBEDDING_MODEL_NAME]

    if not chat_ollama_models:
        st.warning(f"No chat-capable Ollama models found. Please pull some models (e.g., 'ollama pull llama3') and restart the app. Note: '{EMBEDDING_MODEL_NAME}' is for embeddings only.")
        selected_model_id = None
    else:
        selected_model_id = st.sidebar.selectbox(
            "Select Local Ollama Model (for chat):",
            chat_ollama_models,
            help=f"Choose an LLM model available on your local Ollama instance for chat. Note: '{EMBEDDING_MODEL_NAME}' is reserved for RAG embeddings."
        )

# --- IBM Cloud Information Lookup Section (Dynamic) ---
if target == "IBM Cloud":
    st.sidebar.header("IBM Cloud Info Lookup (Dynamic)")
    st.sidebar.markdown("Dynamically fetch LLMs for selected IBM Cloud regions (requires valid IBM Cloud token).")

    selected_ibm_region_code = st.sidebar.selectbox(
        "Select IBM Cloud Region:",
        options=list(IBM_WATSONX_AI_REGIONS.keys()),
        format_func=lambda x: f"{IBM_WATSONX_AI_REGIONS[x]} ({x})",
        help="Select an IBM Cloud region to dynamically fetch available LLMs in Watsonx.ai."
    )

    # Placeholder for displaying LLM list results in the main area
    ibm_llm_list_placeholder = st.empty()

    if st.sidebar.button("Get IBM Cloud Region LLMs"):
        if selected_ibm_region_code and ibm_cloud_authorization and ibm_cloud_authorization != "YOUR_ACCESS_TOKEN":
            with st.spinner(f"Fetching LLMs for {IBM_WATSONX_AI_REGIONS[selected_ibm_region_code]} ({selected_ibm_region_code})..."):
                dynamic_llms = get_ibm_watsonx_ai_llms(
                    selected_ibm_region_code,
                    ibm_cloud_authorization,
                    ibm_cloud_project_id
                )

                with ibm_llm_list_placeholder.container():
                    st.subheader(f"Available LLMs in Watsonx.ai for {IBM_WATSONX_AI_REGIONS[selected_ibm_region_code]} ({selected_ibm_region_code}):")
                    if dynamic_llms:
                        for llm in dynamic_llms:
                            st.write(f"- {llm}")
                    else:
                        st.info("No LLMs found or an error occurred. Please check your IBM Cloud API token and permissions.")

                    st.info(
                        "**Note on Data Centers:** While LLMs are hosted in specific regions, directly listing "
                        "the exact physical data centers associated with each LLM via a public API is not typically feasible. "
                        "Regions (like 'us-south') are backed by multiple data centers, and IBM manages the underlying infrastructure."
                    )
        else:
            st.sidebar.warning("Please select an IBM Cloud region and ensure your IBM Cloud 'Authorization Header' is correctly configured with a valid IAM token.")


# --- RAG Section ---
st.sidebar.header("Retrieval Augmented Generation (RAG)")
enable_rag = st.sidebar.checkbox("Enable RAG", value=False, help="Use RAG to augment LLM responses with information from your documents.")

if enable_rag:
    st.sidebar.subheader("Milvus Connection")
    milvus_status_message, milvus_connected = test_milvus_connection()
    if milvus_connected:
        st.sidebar.success(milvus_status_message)
    else:
        st.sidebar.error(milvus_status_message)

    st.sidebar.info(
        "**Milvus Persistence:** Your Milvus collection is designed to be persistent. "
        "If you're using Docker, ensure you've configured a persistent volume to avoid data loss on container restarts. "
        "This application does not automatically delete your Milvus collection."
    )

    st.sidebar.subheader("Upload Documents for RAG")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents (.txt, .json, .md)", # Reverted file types
        type=["txt", "json", "md"], # Reverted file types
        accept_multiple_files=True,
        help="Upload text, JSON, or Markdown files to build your knowledge base for RAG."
    )

    if st.sidebar.button("Process Documents"):
        if not milvus_connected:
            st.sidebar.error("Cannot process documents: Milvus is not connected. Please ensure Milvus is running.")
        elif not uploaded_files:
            st.sidebar.warning("Please upload at least one document to process.")
        else:
            with st.spinner("Processing documents and inserting into Milvus... This may take a while."):
                # Check current entity count to determine if it's a new or existing collection
                connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
                current_entities = 0
                if utility.has_collection(MILVUS_COLLECTION_NAME):
                    collection_check = Collection(MILVUS_COLLECTION_NAME)
                    current_entities = collection_check.num_entities
                connections.disconnect("default")

                if current_entities == 0:
                    st.sidebar.info("Populating a new Milvus knowledge base.")
                else:
                    st.sidebar.info(f"Adding/updating documents to existing Milvus knowledge base (currently has {current_entities} entities).")


                all_chunks = []
                for uploaded_file in uploaded_files:
                    file_content = ""
                    try:
                        file_mime_type = uploaded_file.type
                        file_extension = uploaded_file.name.split('.')[-1].lower()

                        if file_mime_type == "application/json":
                            json_data = json.load(uploaded_file)
                            file_content = json.dumps(json_data, indent=2)
                            st.sidebar.info(f"Loaded JSON from '{uploaded_file.name}'. Converting to text for chunking.")
                        elif file_mime_type in ["text/markdown", "text/plain"]:
                            file_content = uploaded_file.read().decode("utf-8")
                            st.sidebar.info(f"Loaded text from '{uploaded_file.name}'.")
                        else:
                            st.sidebar.warning(f"Unsupported file type for '{uploaded_file.name}': {file_mime_type}. Skipping.")
                            continue # Skip to next file

                        if file_content: # Only chunk if content was successfully extracted
                            chunks = chunk_text(file_content)
                            all_chunks.extend(chunks)
                            st.sidebar.info(f"Processed {len(chunks)} chunks from '{uploaded_file.name}'.")
                        else:
                            st.sidebar.warning(f"No content extracted from '{uploaded_file.name}'. Skipping.")

                    except json.JSONDecodeError:
                        st.sidebar.error(f"Error decoding JSON from '{uploaded_file.name}'. Please ensure it's valid JSON.")
                    except Exception as e:
                        st.sidebar.error(f"Error reading file '{uploaded_file.name}': {e}")
                
                if all_chunks:
                    st.sidebar.info(f"Generating embeddings for {len(all_chunks)} chunks using '{EMBEDDING_MODEL_NAME}'...")
                    embeddings = embed_text_ollama(all_chunks)

                    if embeddings and len(embeddings) == len(all_chunks):
                        documents_to_insert = {
                            "embeddings": embeddings,
                            "texts": all_chunks
                        }
                        success, message = insert_documents_into_milvus(documents_to_insert)
                        if success:
                            st.sidebar.success(message)
                            # Display how many entities are now in the collection
                            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
                            collection = Collection(MILVUS_COLLECTION_NAME)
                            st.sidebar.info(f"Milvus collection now contains {collection.num_entities} entities.")
                            connections.disconnect("default")
                            milvus_status_message, milvus_connected = test_milvus_connection()
                            st.sidebar.success(milvus_status_message)
                        else:
                            st.sidebar.error(message)
                    else:
                        st.sidebar.error("Failed to generate embeddings for all chunks. Check Ollama server and embedding model.")
                else:
                    st.sidebar.warning("No text chunks extracted from uploaded files.")

    # Add button to clear Milvus collection
    if st.sidebar.button("Clear Milvus Collection", help="Deletes all data from the Milvus collection."):
        if milvus_connected:
            # Use a confirmation dialog to prevent accidental deletion
            if st.sidebar.checkbox("Confirm clearing Milvus collection?", key="confirm_clear_milvus"):
                try:
                    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
                    if utility.has_collection(MILVUS_COLLECTION_NAME):
                        utility.drop_collection(MILVUS_COLLECTION_NAME)
                        st.sidebar.success(f"Milvus collection '{MILVUS_COLLECTION_NAME}' cleared successfully.")
                    else:
                        st.sidebar.info(f"Milvus collection '{MILVUS_COLLECTION_NAME}' does not exist, nothing to clear.")
                    connections.disconnect("default")
                    st.rerun() # Rerun to update status messages
                except Exception as e:
                    st.sidebar.error(f"Error clearing Milvus collection: {e}")
            else:
                st.sidebar.info("Please confirm to clear Milvus collection.")
        else:
            st.sidebar.warning("Milvus is not connected, cannot clear collection.")

    # Slider for top_k in Milvus search
    milvus_top_k = st.sidebar.slider(
        "Number of RAG Documents to Retrieve (Top K)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Controls how many most relevant document chunks are retrieved from Milvus for RAG."
    )


# --- Common LLM Parameters ---
st.sidebar.header("LLM Generation Parameters")
frequency_penalty = st.sidebar.number_input(
    "Frequency Penalty",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.01,
    help="Penalizes new tokens based on their existing frequency in the text, reducing repetition."
)
min_tokens = st.sidebar.number_input(
    "Min Tokens",
    min_value=1,
    value=1,
    step=1,
    help="The minimum number of tokens to generate."
)
max_tokens = st.sidebar.number_input(
    "Max Tokens",
    min_value=1,
    value=256,
    step=1,
    help="The maximum number of tokens to generate."
)
presence_penalty = st.sidebar.number_input(
    "Presence Penalty",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.01,
    help="Penalizes new tokens based on whether they appear in the text, encouraging new topics."
)
temperature = st.sidebar.number_input(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.01,
    help="Controls the randomness of the output. Higher values mean more creative/random."
)
top_p = st.sidebar.number_input(
    "Top P",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.01,
    help="Nucleus sampling: model considers tokens with top_p probability mass."
)
seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    value=0,
    step=1,
    help="Sets the random seed for reproducible results. Set to 0 for no fixed seed."
)
stop = st.sidebar.text_input(
    "Stop Sequences (comma-separated)",
    value="",
    help="Up to 4 sequences where the API will stop generating further tokens."
)
stop_sequences = [s.strip() for s in stop.split(',') if s.strip()] if stop else []

# --- Ollama Model Management Section ---
st.sidebar.header("Ollama Model Management")
st.sidebar.markdown("Explore and manage your local Ollama models.")
st.sidebar.link_button("Search Ollama Models Online", "https://ollama.com/search", help="Opens the official Ollama model search page in a new tab.")

ollama_command_input = st.sidebar.text_input(
    "Ollama Command (e.g., 'ollama run llama3' or 'ollama pull nomic-embed-text')",
    value="ollama ",
    key="ollama_run_command",
    help="Enter a full Ollama command (e.g., 'ollama run <model_name>' or 'ollama pull <model_name>')."
)

ollama_output_placeholder = st.sidebar.empty()

if st.sidebar.button("Execute Ollama Command"):
    if ollama_command_input and (ollama_command_input.strip().startswith("ollama run ") or \
                                 ollama_command_input.strip().startswith("ollama pull ")):
        command_to_execute = ollama_command_input.strip()
        ollama_output_placeholder.info(f"Executing: `{command_to_execute}`")
        
        try:
            process = subprocess.Popen(
                command_to_execute,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            full_output = []
            output_container = ollama_output_placeholder.container()
            output_text_area = output_container.code("", language="bash")

            for line in process.stdout:
                full_output.append(line)
                output_text_area.code("".join(full_output), language="bash")

            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                ollama_output_placeholder.error(f"Command exited with error code: {return_code}")
            else:
                ollama_output_placeholder.success("Command executed successfully. Refreshing model list...")
                st.rerun()
        except FileNotFoundError:
            ollama_output_placeholder.error("Ollama command not found. Ensure Ollama is installed and in your system's PATH.")
        except Exception as e:
            ollama_output_placeholder.error(f"An error occurred during command execution: {e}")
    else:
        st.sidebar.warning("Please enter a valid Ollama command (e.g., 'ollama run llama3' or 'ollama pull nomic-embed-text') to execute.")


# --- Main content area for interaction ---
st.header("LLM Interaction")
user_prompt = st.text_area("Enter your prompt here:", height=150)

if st.button("Generate Response"):
    if not user_prompt:
        st.warning("Please enter a prompt to generate a response.")
    else:
        with st.spinner("Generating response..."):
            final_prompt = user_prompt
            retrieved_chunks_display = [] # To store chunks for display
            if enable_rag:
                milvus_status_message, milvus_connected = test_milvus_connection()
                if not milvus_connected:
                    st.error("RAG is enabled but Milvus is not connected. Please ensure Milvus is running and accessible.")
                    st.stop()
                
                try:
                    ollama_client = get_ollama_client()
                    # Embed the user's query using the EMBEDDING_MODEL_NAME
                    query_embedding_response = ollama_client.embeddings(model=EMBEDDING_MODEL_NAME, prompt=user_prompt)
                    query_embedding = query_embedding_response['embedding']

                    # Search Milvus for relevant chunks
                    # Use the milvus_top_k slider value
                    retrieved_chunks = search_milvus(query_embedding, top_k=milvus_top_k)
                    retrieved_chunks_display = retrieved_chunks # Store for display

                    if retrieved_chunks:
                        context = "\n\n".join(retrieved_chunks)
                        # Explicitly instruct the LLM to handle JSON within the context
                        final_prompt = (
                            "You are provided with context that may contain JSON data. "
                            "Your task is to extract specific information, such as API keys, "
                            "from this JSON context if it is directly relevant to the question. "
                            "If the requested information (e.g., a specific key's value) is found, "
                            "provide only that value. Otherwise, state clearly that the information "
                            "is not available in the provided context.\n\n"
                            f"Context: {context}\n\nQuestion: {user_prompt}"
                        )
                        st.info("RAG: Augmented prompt with retrieved context and JSON extraction instruction.")
                        st.markdown("**Augmented Prompt:**")
                        st.code(final_prompt)
                    else:
                        st.info("RAG: No relevant documents found in Milvus. Proceeding with original prompt.")
                except Exception as e:
                    st.error(f"RAG Error during retrieval: {e}. Proceeding with original prompt.")
                    final_prompt = user_prompt

            # Display retrieved chunks if RAG is enabled and chunks were found
            if enable_rag and retrieved_chunks_display:
                st.subheader("Retrieved RAG Chunks:")
                for i, chunk in enumerate(retrieved_chunks_display):
                    st.text_area(f"Chunk {i+1}", value=chunk, height=100, key=f"retrieved_chunk_{i}")
                st.markdown("---") # Separator

            try:
                if target == "Local Ollama":
                    if not selected_model_id:
                        st.error("No local Ollama model selected. Please select one from the sidebar.")
                    else:
                        st.info(f"Generating response using **{selected_model_id}** (Local Ollama).") # Confirmation message
                        messages = [{"role": "user", "content": final_prompt}]

                        ollama_options = {
                            "temperature": temperature,
                            "top_p": top_p,
                            "seed": seed,
                            "num_predict": max_tokens,
                        }
                        if frequency_penalty > 0 or presence_penalty > 0:
                            ollama_options["repeat_penalty"] = 1.0 + frequency_penalty + presence_penalty
                        if stop_sequences:
                            ollama_options["stop"] = stop_sequences

                        response_generator = ollama.chat(
                            model=selected_model_id, # Use selected_model_id for chat
                            messages=messages,
                            options=ollama_options,
                            stream=True
                        )

                        full_response = ""
                        response_container = st.empty()
                        for chunk in response_generator:
                            if "content" in chunk["message"]:
                                full_response += chunk["message"]["content"]
                                response_container.markdown(full_response + "▌")
                        response_container.markdown(full_response)

                elif target == "IBM Cloud":
                    if not ibm_cloud_llm_model_name or ibm_cloud_project_id == "YOUR_IBM_CLOUD_PROJECT_ID" or ibm_cloud_authorization == "YOUR_ACCESS_TOKEN":
                        st.error("Please configure IBM Cloud settings in the sidebar (Model Name, Project ID, and Authorization Token).")
                    else:
                        st.info(f"Generating response using **IBM Cloud LLM ({ibm_cloud_llm_model_name})**.") # Confirmation message
                        headers = {
                            "Authorization": f"Bearer {ibm_cloud_authorization}",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                        payload = {
                            "messages": [{"role":"user","content":final_prompt}],
                            "project_id": ibm_cloud_project_id,
                            "model_id": ibm_cloud_llm_model_name,
                            "parameters": {
                                "decoding_method": "greedy" if temperature == 0 else "sample",
                                "min_new_tokens": min_tokens,
                                "max_new_tokens": max_tokens,
                                "repetition_penalty": 1.0 + frequency_penalty + presence_penalty,
                                "temperature": temperature,
                                "top_p": top_p,
                                "random_seed": seed,
                                "stop_sequences": stop_sequences,
                            }
                        }
                        
                        if ibm_cloud_version:
                            payload["parameters"]["version"] = ibm_cloud_version

                        response = requests.post(ibm_cloud_url, headers=headers, json=payload)
                        response.raise_for_status()

                        ibm_response_data = response.json()
                        if ibm_response_data and "results" in ibm_response_data and ibm_response_data["results"]:
                            generated_text = ibm_response_data["results"][0].get("generated_text", "No text generated.")
                            st.markdown(generated_text)
                        elif ibm_response_data and "choices" in ibm_response_data and ibm_response_data["choices"]:
                            generated_text = ibm_response_data["choices"][0].get("message", {}).get("content", "No text generated.")
                            st.markdown(generated_text)
                        else:
                            st.warning("IBM Cloud LLM did not return a generated text. Response structure might be unexpected.")
                            st.json(ibm_response_data)

            except requests.exceptions.RequestException as req_err:
                st.error(f"Network or API Error: {req_err}. Check your URL, network connection, and API key/token.")
            except json.JSONDecodeError:
                st.error("Failed to decode JSON response from the API. Check the API endpoint and response format.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

