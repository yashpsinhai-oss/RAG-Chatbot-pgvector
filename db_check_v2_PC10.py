#postgres connection,embedding a vector and reading back from it
import psycopg
import os

from embed_demo_v3_PC9 import get_embedding

from dotenv import load_dotenv
load_dotenv()

if __name__=="__main__":
    sentence="Adding an index on the email column cut our search time from seconds to milliseconds."
    vector=get_embedding(sentence)
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:       # like DriverManager.getConnection(url); "with" closes it for you
        row = conn.execute("SELECT version();").fetchone()       # like a Statement + ResultSet; fetchone() returns one row as a tuple
        print(row)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        result=conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';").fetchone()
        if result is None:
            raise TypeError("Extensions are not installed")
        print(result)
        conn.execute("CREATE TABLE IF NOT EXISTS items(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,content TEXT NOT NULL,embedding VECTOR(768))")
        conn.execute("INSERT INTO items (content,embedding) VALUES (%s,%s::vector)", (sentence,str(vector)))   # %s placeholders are like ? in a PreparedStatement,# %s::vector casts the text into a pgvector value, # str(list) gives "[-0.03, 0.01, ...]", which is the text format pgvector accepts
        resultset=conn.execute("SELECT id,content,vector_dims(embedding) FROM items").fetchall() # vector_dims() asks Postgres for the length, since Python would only see a string, #fetchall- a list of tuples, one per row
        print(resultset)
    # leaving the "with" block commits the transaction if nothing failed, and rolls back if an exception occurred
