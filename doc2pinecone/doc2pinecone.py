# langchain to prep documents for usage

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/paulpoledna/chatgpt-retrieval-plugin/doc2pinecone/rtdocs/wimsatt.pdf")
docs = loader.load()
len (docs)

# print(docs[100].page_content)

# tiktoken to first tokenize docs

import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

from tqdm.auto import tqdm

documents = []

for doc in tqdm(docs):
    chunks = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        documents.append({
            'text': chunk,
        })

data = [
    {
        'text': chunk,
    } for i, chunk in enumerate(chunks)
]

len(documents)
