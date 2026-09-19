from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

# 1. Force the stable 'v1' API version.
# 2. Use the versioned model string 'gemini-1.5-flash-latest' or 'gemini-1.5-pro'.
# 3. We avoid the 'transport' argument which is deprecated in the new SDK.
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


result = model.invoke("what is the capital of india")

print(result)
print(result.content)