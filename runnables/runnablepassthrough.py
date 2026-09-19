from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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

create_joke=RunnableSequence(prompt1,model,parser)

branches=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explaination":RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(create_joke,branches)

result=final_chain.invoke({"topic":"pm of india"})

print("the joke is :",result["joke"])
print("the explaination is :",result["explaination"])