#!/usr/bin/env python3

# Following code sample heavily modified from https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c

from bs4 import BeautifulSoup
from getpass import getpass
from googlesearch import search
import os
from pathlib import Path
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatCohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import HumanMessage, SystemMessage


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
    query = "Who won the last World Series of baseball, and in what year?"
    llm = ChatCohere()
    print(f"Sending non-RAG augmented generation request for query: {query}")
    message = [
        HumanMessage(
            content=query
        )
    ]
    result = llm(message)
    print(f"Result: {result}")

    print(f"Now retrieve results from a Google web search for the same query")
    result_text = ""
    for result_url in search(query, tld="com", num=10, stop=10, pause=2):
        response = requests.get(result_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        result_text = result_text + soup.get_text()

    # Split the result text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(result_text)
    print(f"*** Number of text chunks: {len(texts)}")
    #print(f"Contents of first text chunk: {texts[0]}")

    # Define Embeddings Model
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    print(f"*** Setting up the embeddings model...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    # Set up the base vector store retriever
    print(f"*** Setting up the base vector store retriever")
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
    docs = retriever.get_relevant_documents(query)
    #pretty_print_docs(docs)

    # Applying Reranking with CohereRerank
    print(f"*** Applying re-ranking with CohereRerank")
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compressed_docs)

    # The Generation part of RAG Pipeline
    print(f"*** Now do the RAG generation with query: {query}")
    qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=retriever)
    print(f"*** Running generation: {qa.run(query=query)}")


if __name__ == "__main__":
    main()

