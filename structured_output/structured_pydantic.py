from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Review(BaseModel):
    summary:str=Field(description="Point out the key points from the review")
    sentiment:int=Field(gt=0,lt=10,description="rate the sentiment abou the review out of 10")



structured_model=model.with_structured_output(Review)

result=structured_model.invoke(""" Kamakhya View Point – A Panoramic Gem Overlooking Guwahati

Perched near the revered Kamakhya Temple, the Kamakhya View Point offers an unparalleled panoramic vista of Guwahati. From this elevated spot, visitors can witness the expansive cityscape, with the majestic Brahmaputra River weaving through the urban tapestry. The view is especially enchanting during sunrise and sunset, as the sky paints the city in hues of gold and crimson.

The viewpoint is easily accessible, making it a convenient stop for both pilgrims visiting the temple and tourists exploring the city. While the area can get crowded during peak hours, especially during festivals, the serene ambiance and breathtaking scenery make the visit worthwhile. It’s advisable to carry water and wear comfortable footwear, as the terrain can be uneven in places.

For photography enthusiasts, the Kamakhya View Point provides a perfect backdrop to capture the essence of Guwahati. Whether you’re seeking a moment of reflection or aiming to capture the city’s beauty, this spot offers a memorable experience.""")

print(result)

