#postgres connection
import psycopg
import os

from dotenv import load_dotenv
load_dotenv()
with psycopg.connect(os.getenv("DATABASE_URL")) as conn:       # like DriverManager.getConnection(url); "with" closes it for you
    row = conn.execute("SELECT version();").fetchone()       # like a Statement + ResultSet; fetchone() returns one row as a tuple
    print(row)
    conn.execute("INSERT INTO items (content) VALUES (%s)", ("hello",))   # %s placeholders are like ? in a PreparedStatement
# leaving the "with" block commits the transaction if nothing failed, and rolls back if an exception occurred
    