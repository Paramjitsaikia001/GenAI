from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1=PromptTemplate(
    template="give me a joke on {topic}",
    input_variables=["topic"]
)



parser=StrOutputParser()

create_joke=RunnableSequence(prompt1,model,parser)

#function to return the no. of word
def word_C(text):
    return len(text.split())


branches=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_C)
})

final_chain=RunnableSequence(create_joke,branches)

result=final_chain.invoke({"topic":"pm of india"})

print("the joke is :",result["joke"])
print("the explaination is :",result["word_count"])