#!/usr/bin/env python3

# Following code sample modified from https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c

from getpass import getpass
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
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
    if not os.path.exists("source-materials"):
        os.mkdir("source-materials")

    # Start with making a generation request without RAG augmentation
    query = "What is Vector Institute doing to address AI safety and trustworthiness?"
    llm = ChatCohere()
    print(f"Sending non-RAG augmented generation request for query: {query}")
    message = [
        HumanMessage(
            content=query
        )
    ]
    result = llm(message)
    print(f"Result: {result}")

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

    # Applying Reranking with CohereRerank
    # TODO: Why are we reranking after running the RAG query? Shouldn't we be doing it before?
    print(f"*** Applying re-ranking with CohereRerank")
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compressed_docs)


if __name__ == "__main__":
    main()

