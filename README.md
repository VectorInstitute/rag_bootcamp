# RAG Bootcamp

This is a collection of reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, scheduled to take place in February-March 2024. It demonstrates some of the common methodologies used in RAG workflows (data ingestion, chunks, embeddings, vector databases, sparse/dense retrieval, reranking) using the popular Python [langchain](https://python.langchain.com/docs/get_started/introduction) and [llama_index](https://docs.llamaindex.ai/en/stable/) libraries.

These implementations focus specifically on RAG workflows using [Cohere](https://cohere.com/), for which you will need a free tier API access key. However they are also meant to be used as templates. Using the rich feature set provided by both langchain and llama_index, it should be straightforward to modify these examples to use many other common LLM services.
 

## Git Repostory

Start by cloning this git repository to a local folder:

```
git clone https://github.com/VectorInstitute/rag_bootcamp
```

## (Optional) Build the Python environments
 
**These instructions only apply if you are not running this code on the Vector Institute cluster.** If you are are working on the Vector cluster, these environments are already installed and ready to use.

Requirements:
 - Python 3.10

The notebooks contained in this repository depending on several different Python environments. Build these environments using the following instructions: 

```
cd envs
python3 --version # Make sure this shows Python 3.10!

# Install the dataloaders environment
python3 -m venv ./rag_dataloaders
source rag_dataloaders/bin/activate
python3 -m pip install -r rag_dataloaders/requirements.txt
deactivate

# Install the pubmed_qa environment
python3 -m venv ./rag_local
source rag_local/bin/activate
python3 -m pip install -r rag_local/requirements.txt
deactivate

# Install the pubmed_qa environment
python3 -m venv ./rag_pubmed_qa
source rag_pubmed_qa/bin/activate
python3 -m pip install -r rag_pubmed_qa/requirements.txt
deactivate
```

## Add the Jupyter notebook kernels

These kernels are required for the notebooks in this repository. You can make them available to Jupyter with the following instructions:

```
cd envs

source rag_dataloaders/bin/activate
ipython kernel install --user --name=rag_dataloaders
deactivate

source rag_pubmed_qa/bin/activate
ipython kernel install --user --name=rag_pubmed_qa
deactivate
```

## Reference Implementations

This repository includes several reference implementations showing different approaches and methodologies related to Retrieval-Augmented Generation.

### Ingestion 

These are focused on ingesting data into a RAG pipeline from several commonly-used sources:

- **Web Search**: Popular LLMs like Cohere and OpenAI are very good at processing natural language, but their knowledge is limited by the data they were trained on. As of January 2024, neither service can correctly answer the question "Who won the 2023 World Series of baseball?"
- **Document Search**: Use a collection of unstructured documents to answer domain-specific questions, like: "How many graudate students did Vector Institute sponsor in 2022?"
- **SQL Search**: Answer natural language questions with information from structured relational data
- **Cloud Search**: Retrieve information from data in a cloud service, in this example AWS S3 storage
