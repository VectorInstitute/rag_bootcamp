# RAG Bootcamp: SQL Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place in February-March 2024. Popular LLMs like Cohere and OpenAI are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we answer natural language questions with information from SQL structured relational data. This uses a financial dataset from a Portugese banking instituation, [available on Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets), stored in a sqlite3 database.

## Requirements

* Python 3.10+
* Cohere API key saved in your home directory at `~/.cohere.key`