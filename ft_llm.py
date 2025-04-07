from langchain_chroma import Chroma
from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings

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

context = list(map(lambda document: document.page_content, context))
prompt = f"""
<|im_start|>system
Ты гейм-мастер ролевой игры. Тебе даны начало сюжета пользователя и контекст. Напиши продолжение истории длиной 3-5 предложений пользуясь этими данными.
<|im_end|>
<|im_start|>user
Сюжет: {query}
<|im_end|>
<|im_start|>system
Контекст: 
{'\n---\n'.join(context)}
<|im_end|>
<|im_start|>system
Напиши продолжение:
<|im_end|>""".strip()


max_seq_length = 4096
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-rouling",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")


output = model.generate(**inputs, max_new_tokens=512)
output = tokenizer.batch_decode(output)
output = output[0]
print(output)
