import os
import re
import unicodedata
from pypdf import PdfReader
# __file__ is the path of THIS script file; each file gets its own value
# abspath() turns it into a full path like C:\Users\ADMIN\Downloads\PythonProjects\rag-chatbot-pgvector\chunk_PC11.py
# dirname() drops the file name, leaving the folder that contains the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# join() adds data\jwt_notes.pdf with the right separators for your OS
DOC_PATH = os.path.join(BASE_DIR, "data", "jwt_notes.pdf")

#chunking script for a document, first we need to open/read the document
def load_document(path):
    reader = PdfReader(path)                  # opens the PDF;

    pages = []
    for page in reader.pages:                                 # reader.pages is a list-like of page objects
        pages.append(page.extract_text() or "")               # extract_text() returns the page's text; "or" turns None into ""

    text = "\n".join(pages)                                   # one string for the whole document, ready for chunking

    return text

#cleaning
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)         # fuses like "ﬀ" become "ff"; do this first
    text = re.sub(r"[ \t]+", " ", text)                # runs of spaces and tabs become one space
    text = re.sub(r" *\n *", "\n", text)               # strip spaces around each newline, so "blank" lines are really empty
    text = re.sub(r"\n{2,}", "\n\n", text)             # collapse several blank lines into one paragraph break
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # a lone newline inside a paragraph becomes a space

    return text

def chunk_words(text, size=150, overlap=30):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()                      # splits on any whitespace, giving a list of words
    step = size - overlap                     # 70: how far the window jumps each time
    chunks = []                               # the finished chunks go here

    for i in range(0, len(words), step):      # i = index of the first word of this chunk: 0, 70, 140, ...
        window = words[i:i + size]            # up to `size` words starting at i; the slice stops safely at the end of the list
        chunks.append(" ".join(window))       # glue the words back into one string with single spaces

        if i + size >= len(words):            # this window already reached the last word,
            break                             # so stop; otherwise the next window would hold only repeated overlap words

    return chunks

def chunk_text(text,size=500,overlap=50):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks = []                               # the finished chunks go here
    for i in range(0, len(text), step):      # i = index of the first character of this chunk: 0, 450, 900, ...
        window = text[i:i + size]             # up to `size` characters starting at i; the slice stops safely at the end of the string
        chunks.append(window)                 # add this chunk to the list

        if i + size >= len(text):             # this window already reached the last character,
            break                             # so stop; otherwise the next window would hold only repeated overlap characters

    return chunks

def prepare_text(path, start_marker=None):
    text = clean_text(load_document(path))     # load, then clean, in one place
    if start_marker:                           # optional: skip front matter before a known heading
        idx = text.find(start_marker)          # position of the marker, or -1 if it isn't there
        if idx != -1:                          # only slice when the marker was found, so -1 can't chop the last character
            text = text[idx:]
    return text

def print_stats(chunks):
    if not chunks:
        raise ValueError("No chunks produced")
    print(f"Document has {len(chunks)} chunks")
    max_len=max(len(chunk) for chunk in chunks)
    print(f"Longest chunk has {max_len} characters")
    min_len=min(len(chunk) for chunk in chunks)
    print(f"Shortest chunk has {min_len} characters")
    print(f"First chunk:\n{chunks[0]}\n")
    print(f"Last chunk:\n{chunks[-1]}\n")
    print(f"Middle chunk:\n{chunks[len(chunks)//2]}\n")

if __name__=="__main__":
    text = prepare_text(DOC_PATH, "1. Why JWT")
    chunks=chunk_text(text,500,50)
    print_stats(chunks)
    chunks=chunk_words(text,150,30)
    print_stats(chunks)
