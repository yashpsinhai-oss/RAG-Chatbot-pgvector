import os
from dotenv import load_dotenv
from google import genai # the SDK package is google-genai, but you import it as google.genai
from google.genai import types # holds the config classes

load_dotenv() # load GEMINI_API_KEY from .env
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # pass the key explicitly, like Groq

result=client.models.embed_content(
    model="gemini-embedding-001", # the embedding model we chose
    contents="I love databases", # a string, or a list of strings to embed several at once
    config=types.EmbedContentConfig(
        output_dimensionality=768, # shrink the vector from the 3072 default to 768-dimension vector
    )
)
if not result.embeddings:                                  # guard: None or empty means the call returned nothing
    raise ValueError("No embeddings returned") 

vector=result.embeddings[0].values # one embedding per input, in the same order; .values is the list of floats
print(len(vector)) # should print 768
print(vector[:5]) # first five numbers, as a sanity check