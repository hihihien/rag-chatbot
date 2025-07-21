from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

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

# === Q&A Loop ===
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower().strip() == "q":
        break

    # Retrieve relevant module content
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print("Answer:\n", result)
