from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text="my name is paramjit saikia"

result=embedding.embed_query(text)
print(str(result))