from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

print("Stage 1: Loading")
PATH = "data/"
loader = DirectoryLoader(
    PATH,
    recursive=True,
    show_progress=True,
    loader_cls=TextLoader,
)

documents = loader.load()

print("Stage 2: Splitting")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
splitted = splitter.split_documents(tqdm(documents))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
)

vector_store = Chroma(
    collection_name="potter-512",
    embedding_function=embeddings,
    persist_directory="./db",
)

print("Stage 3: Saving")
for i in tqdm(range(0, len(splitted), 10000)):
    vector_store.add_documents(documents=splitted[i : min(i + 10000, len(splitted))])
