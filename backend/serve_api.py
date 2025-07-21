from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# === Prompt Template ===
model = OllamaLLM(model="llama3.2")
template = """
Du bist ein akademischer Assistent und hilfst Studierenden, ihre Module im Masterstudiengang Medieninformatik zu verstehen.

Hier sind relevante Auszüge aus dem Modulhandbuch:

{reviews}

Bitte beantworte die folgende Frage ausführlich und auf Deutsch:

{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# === Request body ===
class QuestionRequest(BaseModel):
    question: str

# === POST endpoint ===
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    reviews = retriever.invoke(req.question)
    result = chain.invoke({"reviews": reviews, "question": req.question})
    return {"answer": result}

# Optional root
@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running."}
