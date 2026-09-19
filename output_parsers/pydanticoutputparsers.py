from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Person(BaseModel):
    name:str=Field(description="Name of the Person")
    age:int=Field(description="age of the Person")
    gender:str=Field(description="Gender of the Person")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="generate a name ,age and gender of a frictional  {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({"place":"india"})

print(result)