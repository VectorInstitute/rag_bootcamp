#!/usr/bin/env python3

# Following code sample modified from https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c

import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import HumanMessage


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():

    # Setup the environment
    try:
        os.environ["OPENAI_API_KEY"] = open(Path.home() / ".openai.key", "r").read().strip()
    except Exception as err:
        print(f"Could not read your OpenAI key. Please make sure this is available in plain text under your home directory in ~/.openai.key: {err}")

    if not os.path.exists("source-materials"):
        os.mkdir("source-materials")

    # Start with making a generation request without RAG augmentation
    query = "What is Vector Institute doing to address AI safety and trustworthiness?"
    llm = ChatOpenAI()
    print(f"Sending non-RAG augmented generation request for query: {query}")
    message = [
        HumanMessage(
            content=query
        )
    ]
    result = llm(message)
    print(f"Result: {result.content}")

    # Load the pdf
    pdf_folder_path = "./source-materials"
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    docs = loader.load()
    print(f"*** Number of source materials: {len(docs)}")

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"*** Number of text chunks: {len(texts)}")
    #print(f"Contents of first text chunk: {texts[0]}")

    print(f"*** Setting up the embeddings model...")
    embeddings = OpenAIEmbeddings(model="text-search-ada-doc-001")

    # Set up the base vector store retriever
    print(f"*** Setting up the base vector store retriever")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
    docs = retriever.get_relevant_documents(query)
    #pretty_print_docs(docs)

    # The Generation part of RAG Pipeline
    print(f"*** Now do the RAG generation with query: {query}")
    qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=retriever)
    print(f"*** Running generation: {qa.run(query=query)}")

    # Applying Reranking
    print(f"*** Applying re-ranking")
    embeddings = OpenAIEmbeddings()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compressed_docs)

    # Generation — RAG Pipeline using compressor retriever
    print(f"*** Now doing the RAG generation augmented with reranked results:")
    qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=compression_retriever)
    
    print(qa.run(query=query))

if __name__ == "__main__":
    main()

