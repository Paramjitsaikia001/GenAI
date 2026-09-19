from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


class Feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field("give the sentiment of a feedback")

parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt=PromptTemplate(
    template="generate the sentiment of given feedback \n {feedback} \n {format_instruction}",
    input_variables=["Feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

classifier_chain=prompt | model | parser2

prompt2=PromptTemplate(
    template="write a appropiate response to the positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="write a appropiate response to the negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive",prompt2 | model | parser1),
    (lambda x:x.sentiment=="negative",prompt3 | model | parser1),
    RunnableLambda(lambda x:"Could not determine the sentiment")
)

result=classifier_chain | branch_chain

print(result.invoke({"feedback":"what a beatiful place"}))
result.get_graph().print_ascii()