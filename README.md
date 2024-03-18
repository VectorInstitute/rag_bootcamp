# RAG Bootcamp

This is a collection of reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, scheduled to take place in February-March 2024. It demonstrates some of the common methodologies used in RAG workflows (data ingestion, chunks, embeddings, vector databases, sparse/dense retrieval, reranking) using the popular Python [langchain](https://python.langchain.com/docs/get_started/introduction) and [llama_index](https://docs.llamaindex.ai/en/stable/) libraries.

These implementations focus specifically on RAG workflows using [Cohere](https://cohere.com/), for which you will need a free tier API access key. However they are also meant to be used as templates. Using the rich feature set provided by both langchain and llama_index, it should be straightforward to modify these examples to use many other common LLM services.

## Reference Implementations

This repository includes several reference implementations showing different approaches and methodologies related to Retrieval-Augmented Generation.

- [**Web Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/web_search): Popular LLMs like Cohere and OpenAI are very good at processing natural language, but their knowledge is limited by the data they were trained on. As of January 2024, neither service can correctly answer the question "Who won the 2023 World Series of Baseball?"
- [**Document Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/document_search): Use a collection of unstructured documents to answer domain-specific questions, like: "How many AI scholarships did Vector Institute award in 2022?"
- [**SQL Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/sql_search): Answer natural language questions with information from structured relational data. This demo uses a financial dataset from a Portugese banking instituation, [available on Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [**Cloud Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/cloud_search): Retrieve information from data in a cloud service, in this example AWS S3 storage
- [**Local Llama2**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/local_llama2): Use an on-prem, fully open-source and locally hosted Llama2-7B model to run a full RAG workflow for document search and retrieval
- [**PubMed QA**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/pubmed_qa): A full pipeline on the [PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) dataset demonstrating ingestion, embeddings, vector index/storage, retrieval, reranking, with a focus on evaluation metrics.
- [**RAG Evaluation**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/rag_evaluation): RAG evaluation techniques based on the [Ragas](https://github.com/explodinggradients/ragas) framework. Focuses on evaluation "test sets" and how to use these to determine how well a RAG pipeline is actually working.
 
## Requirements

* Python 3.10+

## Git Repostory

Start by cloning this git repository to a local folder:

```
git clone https://github.com/VectorInstitute/rag_bootcamp
```

## (Optional) Build the virtual Python environments

**These instructions only apply if you are not running this code on the Vector Institute cluster.** If you are are working on the Vector cluster, these environments are already pre-compiled and ready to use in the `/ssd003/projects/aieng/public/rag_bootcamp/envs` folder.

The notebooks contained in this repository depend on several different Python environments. Build these environments using the following instructions: 

```
cd rag_bootcamp/envs
python3 --version # Make sure this shows Python 3.10+!

# Install the dataloaders environment
python3 -m venv ./rag_dataloaders
source rag_dataloaders/bin/activate
python3 -m pip install -r rag_dataloaders/requirements.txt
deactivate

# Install the evaluation environment
python3 -m venv ./rag_evaluation
source rag_evaluation/bin/activate
python3 -m pip install -r rag_evaluation/requirements.txt
deactivate

# Install the local environment
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
cd rag_bootcamp/envs

source rag_dataloaders/bin/activate
ipython kernel install --user --name=rag_dataloaders
deactivate

source rag_evaluation/bin/activate
ipython kernel install --user --name=rag_evaluation
deactivate

source rag_local/bin/activate
ipython kernel install --user --name=rag_local
deactivate

source rag_pubmed_qa/bin/activate
ipython kernel install --user --name=rag_pubmed_qa
deactivate
```

## Lastly, start a Jupyter notebook

```
cd rag_bootcamp
source envs/rag_local/bin/activate
jupyter notebook --ip $(hostname --fqdn)
```
