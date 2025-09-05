import logging
import sys
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import json
from dotenv import load_dotenv

# --- Configuration ---
FILE_PATH = "calculus_problems.json"
COLLECTION_NAME = "calculus_collection"
EMBEDDING_MODEL = "text-embedding-004"
QDRANT_PATH = "langchain_qdrant"
VECTOR_SIZE = 768
VECTOR_DISTANCE = Distance.COSINE

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Normalize LaTeX for consistent embedding
def normalize_latex(text):
    """Normalize LaTeX to reduce mismatches."""
    text = re.sub(r"\\cdot\s*", "", text)  # Remove \cdot
    text = re.sub(r"\\{2,}", r"\\", text)  # Fix double backslashes
    text = re.sub(r"\s+", " ", text.strip())  # Normalize spaces
    return text


# Qdrant collection setup
def setup_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int, distance: Distance
):
    """Checks if the Qdrant collection exists and creates it if it doesn't."""
    try:
        if not client.collection_exists(collection_name=collection_name):
            logging.info(
                f"Collection '{collection_name}' does not exist. Creating it now."
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Failed to setup Qdrant collection: {e}")
        raise


# Load and prepare documents
def load_documents(file_path: str):
    """Loads documents from a JSON file and prepares them for embedding."""
    logging.info(f"Loading documents from '{file_path}'.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: Failed to parse JSON from '{file_path}'.")
        raise

    docs = []
    for item in data:
        normalized_problem = normalize_latex(item["problem"])
        doc = Document(
            page_content=normalized_problem,
            metadata={
                "solution": item["solution"],
                "answer": item["answer"],
                "source": "OpenR1-Math-220k",
            },
        )
        docs.append(doc)
    logging.info(f"Loaded {len(docs)} documents.")
    return docs


def main():
    """Main function to run the data ingestion and embedding process."""
    load_dotenv()
    logging.info("Starting the knowledge base ingestion process.")

    try:
        documents = load_documents(FILE_PATH)
        if not documents:
            logging.warning("No documents found or loaded. Exiting.")
            return

        client = QdrantClient(path=QDRANT_PATH)
        setup_qdrant_collection(client, COLLECTION_NAME, VECTOR_SIZE, VECTOR_DISTANCE)

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        logging.info("Adding documents to the vector store...")
        vector_store.add_documents(documents=documents)
        logging.info("Successfully added all documents to the vector store.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
