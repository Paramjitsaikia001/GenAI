# this is outdated now



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

schema = [
    ResponseSchema(name="fact1", description="one good thing"),
    ResponseSchema(name="fact2", description="one bad thing"),
    ResponseSchema(name="fact3", description="one thing from other perspective"),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Write 3 facts about  {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({"topic":"PM of india"})

print(result)