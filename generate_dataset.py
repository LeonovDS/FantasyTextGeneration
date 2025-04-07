from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from random import sample

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
)

vector_store = Chroma(
    collection_name="potter-512",
    embedding_function=embeddings,
    persist_directory="./db",
)

documents = vector_store.get()["documents"]
sample_count = int(len(documents) * 0.1)
prompts = []

for doc in sample(documents, sample_count):
    context = vector_store.similarity_search(doc, k=6)
    context = list(map(lambda document: document.page_content, context))
    context = list(filter(lambda document: document != doc, context))
    head = doc[:100]
    tail = doc[100:]
    prompt = f"""
<|im_start|>system
Ты гейм-мастер ролевой игры. Тебе даны начало сюжета пользователя и контекст. Напиши продолжение истории длиной 3-5 предложений пользуясь этими данными.
<|im_end|>
<|im_start|>user
Сюжет: {head}
<|im_end|>
<|im_start|>system
Контекст: 
{'\n---\n'.join(context)}
<|im_end|>
<|im_start|>system
Напиши продолжение:
<|im_end|>
<|im_start|>assistant
{tail}
<|im_end|>
    """.strip().replace("\n\n", "\n")
    prompts.append(prompt)
    print(prompt)

with open("dataset_1.txt", "w") as out:
    out.write("\n\n".join(prompts))
