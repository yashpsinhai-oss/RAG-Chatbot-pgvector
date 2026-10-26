#error handling, query as user input, embedding sentence and query , then comparing similarity,score, rank in sorted order
import os
import sys
from dotenv import load_dotenv
from google import genai # the SDK package is google-genai, but you import it as google.genai
from google.genai import types # holds the config classes
from google.genai import errors

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

#without batch
def get_embeddings(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001", # the embedding model we chose
        contents=texts, # a string, or a list of strings to embed several at once
        config=types.EmbedContentConfig(
            output_dimensionality=768, # shrink the vector from the 3072 default to 768-dimension vector
        )
    )

    if not result.embeddings:                                  # guard: None or empty means the call returned nothing
        raise ValueError("No embeddings returned") 
    
    vector=result.embeddings[0].values # one embedding per input, in the same order; .values is the list of floats    

    return vector

#with batch
#is the same stride idea as your chunker, without the overlap
#one call could do it, but batching keeps you safe as documents grow
def get_embeddings_batch(texts,batch_size=10):
    vectors=[]

    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        
        result = client.models.embed_content(
            model="gemini-embedding-001", # the embedding model we chose
            contents=batch, # a string, or a list of strings to embed several at once in batches
            config=types.EmbedContentConfig(
                output_dimensionality=768, # shrink the vector from the 3072 default to 768-dimension vector
            )
        )

        if not result.embeddings  or len(result.embeddings)!=len(batch):
            raise ValueError("Embedding count does not match the batch size")  

        vectors.extend(e.values for e in result.embeddings)

    return vectors    

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)

if __name__ == "__main__":
    sentences=["Adding an index on the email column cut our search time from seconds to milliseconds.","Choosing the right data model early saves a lot of pain when the app grows.","Check the index at the back of the cookbook to find the query for kneading dough.","Simmer the onions slowly until they turn golden before adding the spices.","Chicken and rice is a quick weeknight dinner.","The opening batsman scored a century in the final over.","Spin bowlers get more help from a dry, cracked pitch."]
    try:
        vectors = [get_embedding(sentence) for sentence in sentences]
    except errors.ClientError as e:
        print(f"Embedding Failed due to Bad or missing API key, quota or rate limit hit (429), invalid request: {e}")
        sys.exit(1)
    except errors.ServerError as e:
        print(f"Temporary outage, overloaded service: {e}")
        sys.exit(1)
    except errors.APIError as e:
        print(f"Embedding failed due to API error: {e}")
        sys.exit(1)

    while True:
        query_text = input("Query_Txt: ").strip()
        if not query_text:
            continue
        if query_text.lower() in ["exit","quit"]:
            break

        try:
            query_embedding=get_embedding(query_text)
        except errors.ClientError as e:
            print(f"Embedding Failed due to Bad or missing API key, quota or rate limit hit (429), invalid request: {e}")
            continue
        except errors.ServerError as e:
            print(f"Temporary outage, overloaded service: {e}")
            continue
        except errors.APIError as e:
            print(f"Embedding failed due to API error: {e}")
            continue
        
        scores = [cosine_similarity(vector, query_embedding) for vector in vectors]
        
        results=list(zip(scores,sentences))  
        print("Rank"+" "+"Score"+" "+"Sentence")
        for rank, (score,sentence) in enumerate(sorted(results,reverse=True),start=1):
            print(f"{rank} {score:.3f} {sentence}")
        