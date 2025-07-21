import json
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#config files
DATA_FILE = "data/modules.json"
CHROMA_PATH = "chrome_langchain_db"
COLLECTION_NAME = "modulhandbuch_mmi"
EMBED_MODEL = "mxbai-embed-large"

#load data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_modules = json.load(f)

#clean up 
documents = []
ids = []

for i, mod in enumerate(raw_modules):
    # Skip empty or TOC-only entries
    if len(mod["content"]) < 200:
        continue

    doc = Document(
        page_content=mod["content"],
        metadata={
            "module_id": mod["module_id"],
            "title": mod["title"]
        },
        id=str(i)
    )
    documents.append(doc)
    ids.append(str(i))

print(f"Loaded {len(documents)} clean module documents.")

#embedding setup
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

#load vector store
add_documents = not os.path.exists(CHROMA_PATH)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print("Documents embedded and added to Chroma DB.")

#create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
