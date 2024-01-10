#!/usr/bin/env python3

from getpass import getpass
import os
from pathlib import Path

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Cohere
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_experimental.sql import SQLDatabaseChain


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
    #query = "How many employees are at Vector Institute?"
    #query = "How many employees are there?"
    query = "What are the names of all the employees?"
    llm = Cohere()
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    result = llm(query)
    print(f"Result: {result}\n")

    # Now query the database for the information we want
    db = SQLDatabase.from_uri("sqlite:///vector.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    result = db_chain.run(query)
    print(f"The result is: {result}")


if __name__ == "__main__":
    main()

