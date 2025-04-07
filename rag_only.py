from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

query = input("Enter text:")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
)

vector_store = Chroma(
    collection_name="potter-512",
    embedding_function=embeddings,
    persist_directory="./db",
)

context = vector_store.similarity_search(
    query,
    k=5,
)

print(context)

prompt = ChatPromptTemplate(
    messages=[
        (
            "system",
            "Ты гейм-мастер ролевой игры. Тебе даны начало сюжета пользователя и контекст. Напиши продолжение истории длиной 3-5 предложений пользуясь этими данными.",
        ),
        ("user", "Сюжет: {query}"),
        ("system", "Контекст: {context}"),
        ("system", "Напиши продолжение:"),
    ],
)

llm = ChatOllama(model="qwen2.5")
llm.format = None

context = "\nКонтекст: ".join(map(lambda doc: doc.page_content, context))

chain = prompt | llm
print(chain.invoke({"query": query, "context": context}).content)
