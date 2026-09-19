from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model =ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

template1=PromptTemplate(
    template="create 3 question on {topic}",
    input_variables=["topic"]  
)

template2=PromptTemplate(
    template="ans the following questions {q}",
    input_variables=["q"]
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"chain of langchain"})

print(result)
chain.get_graph().print_ascii()