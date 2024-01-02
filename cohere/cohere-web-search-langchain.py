#!/usr/bin/env python3

# Following code sample heavily modified from https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c

from bs4 import BeautifulSoup
from getpass import getpass
from googlesearch import search
import os
from pathlib import Path
import requests

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():

    # Setup the environment
    os.environ["COHERE_API_KEY"] = open(Path.home() / ".cohere.key", "r").read().strip()

    # Start with making a generation request without RAG augmentation
    query = "Who won the 2023 World Series of baseball?"
    llm = Cohere()
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    result = llm(query)
    print(f"Result: {result}\n")

    print(f"*** Now retrieve results from a Google web search for the same query\n")
    result_text = ""
    for result_url in search(query, tld="com", num=10, stop=10, pause=2):
        response = requests.get(result_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        result_text = result_text + soup.get_text()

    # Split the result text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(result_text)
    print(f"*** Number of text chunks: {len(texts)}\n")

    # Define Embeddings Model
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    print(f"*** Setting up the embeddings model...\n")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    # Set up the base vector store retriever
    print(f"*** Setting up the base vector store retriever\n")
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
    docs = retriever.get_relevant_documents(query)
    #pretty_print_docs(docs)

    # Applying Reranking with CohereRerank
    print(f"*** Applying reranking with CohereRerank\n")
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compressed_docs)

    # The Generation part of RAG Pipeline
    print(f"*** Now do the reranked RAG generation with query: {query}\n")
    qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=compression_retriever)
    print(f"*** Running generation...\n")
    print(f"Result: {qa.run(query=query)}")


if __name__ == "__main__":
    main()

