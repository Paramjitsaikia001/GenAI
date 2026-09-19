from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

messeges=[]
'''
without storing conversation ai will cannot get the context of prevous conversation

so we are using this 
but it has also cons that is :
it has no specific marks that it tells you which one is users and ai 

so we will try to avoid this 
'''

history=[
    SystemMessage(content="you are a very rube assistant"),
]

while True:
    prompt=input("User: ")

    # messeges.append(prompt)

    history.append(HumanMessage(content=prompt))
    if prompt=="exit":
        break

    # result=model.invoke(messeges)

    result=model.invoke(history)


    # messeges.append(result.content[0]["text"])

    history.append(AIMessage(result.content[0]["text"]))
    print("AI: ", result.content[0]["text"])

print(messeges)
print(history)
