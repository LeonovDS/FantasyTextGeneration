from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

query = input("Enter text:")

prompt = ChatPromptTemplate(
    messages=[
        (
            "system",
            "Ты гейм-мастер ролевой игры. Тебе дано начало сюжета пользователя. Напиши продолжение истории длиной 3-5 предложений.",
        ),
        ("user", "Сюжет: {query}"),
    ],
)

llm = ChatOllama(model="qwen2.5")
llm.format = None

chain = prompt | llm
print(chain.invoke({"query": query}).content)
