import os
import chromadb
import streamlit as st
import fitz  
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chroma_client = chromadb.PersistentClient(path="chroma_db")
pdf_collection = chroma_client.get_or_create_collection("pdf_data")  
dataset_collection = chroma_client.get_or_create_collection("trained_data")  
memory_collection = chroma_client.get_or_create_collection("chat_memory")  

### PDF TEXT EXTRACTION & EMBEDDINGS

def extract_text_from_pdf(uploaded_file):
    """Extracts and returns text from an uploaded PDF file"""
    try:
        with fitz.open("pdf", uploaded_file.read()) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def store_embeddings(collection, data_dict, model_name="all-MiniLM-L6-v2", batch_size=100):
    """Stores extracted text embeddings into a ChromaDB collection"""
    model = SentenceTransformer(model_name)

    for filename, text in data_dict.items():
        text_chunks = text.split("\n\n")  

        existing_ids = set(collection.get()["ids"] if collection.get() else [])

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            embeddings = model.encode(batch).tolist()
            ids = [f"{filename}_{idx}" for idx in range(i, i+len(batch))]

            new_data = [(id, emb, txt) for id, emb, txt in zip(ids, embeddings, batch) if id not in existing_ids]

            if new_data:
                collection.add(
                    ids=[x[0] for x in new_data],
                    embeddings=[x[1] for x in new_data],
                    metadatas=[{"filename": filename, "text": x[2]} for x in new_data]
                )

    st.success("Data embeddings stored successfully in ChromaDB!")

st.set_page_config(page_title="ESG Chatbot", page_icon="🤖", layout="wide")
st.title("ESG Chatbot")

uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_dict = {file.name: extract_text_from_pdf(file) for file in uploaded_files}
    store_embeddings(pdf_collection, pdf_dict)
    st.success("PDFs uploaded and processed successfully!")

###MEMORY-AWARE RETRIEVAL

def retrieve_documents(collection, query, k=5):
    """Retrieves relevant excerpts from ChromaDB"""
    try:
        results = collection.query(query_texts=[query], n_results=k)
        if results.get("metadatas") and len(results["metadatas"]) > 0:
            return [doc["text"] for doc in results["metadatas"][0] if "text" in doc]
        return []
    except Exception as e:
        return [f"Error retrieving documents: {e}"]

def update_memory(query, response):
    """Stores user queries and responses for context retention"""
    memory_collection.add(
        ids=[f"memory_{len(memory_collection.get()['ids'])}"],
        embeddings=[[0] * 384],  
        metadatas=[{"query": query, "response": response}]
    )

def retrieve_memory(query):
    """Retrieves past responses from memory"""
    try:
        results = memory_collection.query(query_texts=[query], n_results=3)
        if results.get("metadatas"):
            return [doc["response"] for doc in results["metadatas"][0]]
        return []
    except Exception:
        return []

###AI QUERY PROCESSING 

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)

prompt_template = PromptTemplate(
    template="""
    You are an AI assistant with expertise in both trained datasets and uploaded PDFs.

    Query: {query}
    
    Memory Context:
    {memory_context}

    Retrieved PDF Content:
    {pdf_context}

    Retrieved Trained Dataset Content:
    {dataset_context}

    If no relevant information is found, use general knowledge to answer the query.
    """,
    input_variables=["query", "memory_context", "pdf_context", "dataset_context"]
)

def answer_query(query):
    """Processes a user query with memory, trained dataset, PDF retrieval, and LLM"""
    memory_context = "\n".join(retrieve_memory(query))
    pdf_context = "\n".join(retrieve_documents(pdf_collection, query))
    dataset_context = "\n".join(retrieve_documents(dataset_collection, query))

    final_prompt = prompt_template.format(
        query=query,
        memory_context=memory_context,
        pdf_context=pdf_context,
        dataset_context=dataset_context
    )

    response = llm.invoke([SystemMessage(content=final_prompt)]).content
    update_memory(query, response)
    return response



###CHAT INTERFACE

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about your PDFs & trained dataset:")
if query:
    with st.spinner("🔎 Searching..."):
        response = answer_query(query)

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("🤖 AI Bot", response))

for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**🧑‍💼 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")

st.write("💬 **The chatbot remembers previous interactions!**")
