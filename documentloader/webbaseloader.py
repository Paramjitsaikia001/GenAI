from langchain_community.document_loaders import WebBaseLoader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt=PromptTemplate(
    template="Answer the following question \n {question} on topic \n  {topic}",
    input_variables=["question","topic"]
)

parser=StrOutputParser()

url="https://github.com/Paramjitsaikia001"

'''NOTE: we can give list of url too'''
loader=WebBaseLoader(url)

docs=loader.load()


chain=prompt | model | parser

print(chain.invoke({"question":"can you tell me the name of user","topic":docs[0].page_content}))

print(docs)