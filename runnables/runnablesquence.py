from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1=PromptTemplate(
    template="give me a joke on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="explain me the following joke ,\n {joke}",
    input_variables=["joke"]
)

parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({"topic":"tall people"})

print(result)