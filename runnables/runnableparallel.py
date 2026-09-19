from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation",
)

model2=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="write a english tagline of {topic} ",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
   template="write a Hinglish tagline of {topic} ",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain=RunnableParallel(
    {
        "english":RunnableSequence(prompt1,model,parser),
        "hinglish":RunnableSequence(prompt2,model,parser)
    }
)

result=chain.invoke({"topic":"assam"})

print(result)