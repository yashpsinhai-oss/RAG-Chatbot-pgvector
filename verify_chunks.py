import os
import psycopg
from dotenv import load_dotenv
from injest_PC11 import source
load_dotenv()
with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    #count row
    count=conn.execute("SELECT COUNT(*) FROM chunks WHERE source = %s",(source,)).fetchone()[0] # fetchone() returns a tuple like (14,), so [0] pulls out the number
    print(f"Rows stored: {count}") # should equal len(chunks)

    #preview row
    preview=conn.execute("SELECT chunk_index,LEFT(content,60),vector_dims(embedding) "
                         "FROM chunks WHERE source=%s ORDER BY chunk_index LIMIT 5",(source,),).fetchall()     # a list of tuples, one per row
    for row in preview:
        print(row)        # expect (0, '1. Why JWT? The problem...', 768)
