from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

template1=PromptTemplate(
    template="write a brief report about {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="write a tagline to represent the  following content./n {text}",
    input_variables=["text"]
)

#without using strOutputParser
'''
prompt1=template1.invoke({"topic":"black hole"})

result1=model.invoke(prompt1)

prompt2=template2.invoke({"text":result1.content[0]["text"]})

result2=model.invoke(prompt2)

print(result2.content[0]["text"])

'''

#with stroutputParser

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model |  parser

result =chain.invoke({"topic":"PM of india"})

print(result)
