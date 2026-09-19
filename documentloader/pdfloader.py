from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader=PyPDFLoader(
    "textpdf.pdf",
)

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1=PromptTemplate(
    template="summerize the data from  {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="answer the question from the topic .\n question->{question} topic-> {topic}",
    input_variables=["topic","question"]
)

doc=loader.load()

# print(doc)
'''
this is a list and only one element is there but in the one element there is two object one is "source" and other is "page-content"
'''
parser=StrOutputParser()

stored_data=(prompt1 | model | parser).invoke({
    "topic":doc[0].page_content
})
result=( prompt2 | model | parser).invoke({
    "topic":stored_data,
    "question":"what is the tech stack of this person ?"
})

print(result)