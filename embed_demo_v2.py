import os
from dotenv import load_dotenv
from google import genai # the SDK package is google-genai, but you import it as google.genai
from google.genai import types # holds the config classes

load_dotenv() # load GEMINI_API_KEY from .env
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # pass the key explicitly, like Groq

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001", # the embedding model we chose
        contents=text, # a string, or a list of strings to embed several at once
        config=types.EmbedContentConfig(
            output_dimensionality=768, # shrink the vector from the 3072 default to 768-dimension vector
        )
    )

    if not result.embeddings:                                  # guard: None or empty means the call returned nothing
        raise ValueError("No embeddings returned") 

    vector=result.embeddings[0].values # one embedding per input, in the same order; .values is the list of floats    

    return vector

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)

if __name__ == "__main__":
    text1 = "I enjoy SQL"
    text2 = "I love Databases"
    vector1 = get_embedding(text1)
    vector2 = get_embedding(text2)
    # print(vector1) # the embedding vector for the input text
    # print(vector1[:5]) # the embedding vector for the input text, truncated to the first 5 elements
    # print(vector2[:5]) # the embedding vector for the input text, truncated to the first 5 elements
    print(len(vector1)) # should print 768
    print(len(vector2)) # should also print 768
    similarity = cosine_similarity(vector1, vector2)
    print(f"Cosine Similarity: {similarity}")