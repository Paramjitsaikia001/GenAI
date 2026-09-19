from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

messege=[
    SystemMessage(content="you are a sweet helpfull assistant"),
    HumanMessage(content="tell me about langchain")
]

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

result=model.invoke(messege)
messege.append(AIMessage(result.content[0]["text"]))
print(result.content[0]["text"])