#!/usr/bin/env python3

from getpass import getpass
import os
from pathlib import Path

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.chains import create_sql_query_chain
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
    query = "How many people with management jobs applied for a banking deposit in May?"
    llm = Cohere(model="command", max_tokens=256)
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    result = llm(query)
    print(f"Result: {result}\n")

    # Now query the LLM to turn our question into a SQL query
    # The banking_term_deposits data comes from here: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets
    print(f"*** Now translating the question into a SQL query to send to our database...")
    db = SQLDatabase.from_uri("sqlite:///banking_term_deposits.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # BUG: Results get randomly returned with the wrong sql syntax (```sql:...```) and fail. Unfortunately, this is random and seems to be happening on the Cohere side :(
    # The most consistent way to fix this is by modifying the prompt template in /ssd003/projects/aieng/public/rag_bootcamp/rag_venv/lib64/python3.10/site-packages/langchain/chains/sql_database/prompt.py. 
    # On line 215, add the following text to the sqlite prompt: "Do not prepend the SQL Query with ```sql or append it with ```, instead, just prepend it with SQLQuery:"

    # I would prefer to modify the prompt in this example instead of modifying the library code with the following:
    prompt_template = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today". Do not prepend the SQL Query with ```sql or append it with ```, instead, just prepend it with SQLQuery:

Use the following format:

Question: Question here
SQLQuery: SQL Query to run.
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}"""

    # However, this works pretty rarely, not enough to be useful, so let's not use it for now :(
    #db_chain.llm_chain.prompt.template = prompt_template

    result = db_chain.run(query)

    # Ugh, somewhere in the middle here, the result gets dumped back to Cohere again to wrap it up in a LLM generation
    # This usually just returns the answer, but sometimes it muddles it up and even changes the answer returned from the SQL query.

    print(f"Result: {result}")


if __name__ == "__main__":
    main()

