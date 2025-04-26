# Semantic search
# Build a semantic search engine
# https://python.langchain.com/docs/tutorials/retrievers/

import os
import pprint
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain

load_dotenv()

# webpage = "https://docs.smith.langchain.com/user_guide"
webpage = "https://outrival.com/"

loader = WebBaseLoader(webpage)

docs = loader.load()

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

print("\n# Splitting\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Number of splits: {len(all_splits)}")

print("\n# Embedding\n")

project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
print(f"Google Project ID: {project_id}")
embeddings = VertexAIEmbeddings(
    model="text-embedding-004",
    project=project_id,
)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print("Vector 1:", vector_1[:10])

print("\n# Vector store\n")

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

query = "What is OutRival's mission?"
print(f"Query: {query}\n")
results = vector_store.similarity_search_with_score(query)
doc, score = results[0]
print(f"Score: {score}")
print(doc)

print("\n# Retrievers\n")

inputs = [
    "What is OutRival's mission?",
    "When was OutRival founded?",
]
print("inputs:")
pprint.pprint(inputs)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results = retriever.batch(
    [
        "What is OutRival's mission?",
        "Who are the founders of OutRival and what are their backgrounds?",
    ],
)

print("results:")
pprint.pprint(results)
