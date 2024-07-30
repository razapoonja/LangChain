import os

from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "waves_quantum.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def create_vector_store():
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path."
            )

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        print(f"Sample chunk:\n{docs[0].page_content}\n")

        print("\n--- Creating embeddings ---")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
    else:
        print("Vector store already exists. No need to initialize.")

def ask_query(query):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.45},
    )
    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


if __name__ == '__main__':

    create_vector_store()

    query = "Tell me about quantum mechanics"
    result = ask_query(query)

    print(result)
