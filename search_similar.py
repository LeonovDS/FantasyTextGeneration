from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
)

vector_store = Chroma(
    collection_name="text-blocks-1000",
    embedding_function=embeddings,
    persist_directory="./db",
)

ans = vector_store.similarity_search(
    "Фродо взял кольцо в руку",
    k=10,
)
print(" ".join(map(str, ans)))
