from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path="test",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# result=loader.load()

# print(len(result))
# print(result)
# print(result.page_content)


result=loader.lazy_load()

target_page=2
for i , d in enumerate(result,start=1):
    if i==target_page:
        print("metadata of ",i,"is", d.metadata)
        print("page content of ",i,"is", d.page_content)