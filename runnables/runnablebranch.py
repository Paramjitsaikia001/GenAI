from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1=PromptTemplate(
    template="generate a report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="summarize the report  {topic}",
    input_variables=["topic"]
)



parser=StrOutputParser()

report=RunnableSequence(prompt1,model,parser)




branches=RunnableBranch(
   (lambda x:len(x.split())>300,RunnableSequence(prompt2,model,parser)), 
   RunnablePassthrough()
)

final_chain=RunnableSequence(report,branches)

result=final_chain.invoke({"topic":"pm of india"})
print(result)