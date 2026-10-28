#load,clean,embed,chunk,store
from chunking import load_document,clean_text,chunk_text,chunk_words,DOC_PATH,prepare_text
from embeddings import get_embeddings_batch
import psycopg
from dotenv import  load_dotenv
import os
source = os.path.basename(DOC_PATH)                    # just "jwt_notes.pdf", so your Windows username never reaches the database

load_dotenv
if __name__=="__main__":
    text=prepare_text(DOC_PATH,"1. Why JWT")
    chunks=chunk_words(text)
    vectors=get_embeddings_batch(chunks,10)
    
    if len(chunks) != len(vectors):                        # safety check: each chunk must have exactly one vector
        raise ValueError("chunks and vectors differ in length")
    rows = [                                               # one tuple per row to insert, in the same order as the SQL placeholders
        (source, i, chunk, str(vector))                    # str(list) gives "[0.01, -0.03, ...]", the text form pgvector accepts
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))   # i = chunk_index, counting from 0
    ]
    
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:      # one transaction: commits on normal exit, rolls back on any exception
        with conn.cursor() as cur:   
            # cur.execute("DROP TABLE IF EXISTS chunks")                             # executemany lives on a cursor, not on the connection
            cur.execute("CREATE TABLE IF NOT EXISTS chunks(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,source TEXT NOT NULL,chunk_index INT NOT NULL,content TEXT NOT NULL,embedding VECTOR(768),UNIQUE (source,chunk_index))")
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))    # remove old rows for this file; note the comma in (source,)

            cur.executemany(                                      # runs the same INSERT once per tuple in rows
            "INSERT INTO chunks (source, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",                # %s::vector casts the text into a pgvector value
            rows,
        )
# leaving the "with psycopg.connect" block commits the delete and all the inserts together
    print(f"Inserted {len(rows)} chunks for {source}")