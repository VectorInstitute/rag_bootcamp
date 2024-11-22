# RAG Bootcamp: Evaluation Techniques

This notebook presents a RAG workflow for the [PubMed QA](https://pubmedqa.github.io/) task using [LlamaIndex](https://www.llamaindex.ai/). The code is written in a configurable fashion, giving you the flexibility to edit the RAG configuration and observe the change in output/responses. It serves as a good template for various RAG workflows.

## Requirements

1. Python 3.10+
2. If running on the Vector cluster, load the `rag_pubmed_qa` env:
    ```bash
    source /ssd003/projects/aieng/public/rag_bootcamp/envs/rag_pubmed_qa/bin/activate
    ```
   **[Alternate]** Otherwise, create your own env and install packages using the `requirements.txt` file (make sure to run these commands in the top level dir of the repository):
    ```bash
    python3 -m venv ./rag_pubmed_qa
    source rag_pubmed_qa/bin/activate
    python3 -m pip install -r ./envs/rag_pubmed_qa/requirements.txt
    ```
3. Install the jupyter kernel:
    ```bash
    ipython kernel install --user --name=rag_pubmed_qa
    ```
4. If you are accessing the LLMs and embedding models through Vector AI Engineering's **Kaleidoscope Service** (Vector Inference + Autoscaling), you will need to set the KScope env variables:
    - Request a KScope API Key:
        Run the following command (replace `<user_id>` and `<password>`) from **within the cluster** to obtain the API Key. The `access_token` in the output is your KScope API Key.
        ```bash
        curl -X POST -d "grant_type=password" -d "username=<user_id>" -d "password=<password>" https://kscope.vectorinstitute.ai/token
        ```
    - After obtaining the `.env` configurations, make sure to create the `.kscope.env` file in your home directory (`/h/<user_id>`) and set the following env variables in that file:
        ```bash
        export OPENAI_BASE_URL="https://kscope.vectorinstitute.ai/v1"
        export OPENAI_API_KEY=<kscope_api_key>
        ```
   **[Alternate]** If you using **OpenAI models** instead, set the env variables to the following values:
    ```bash
    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export OPENAI_API_KEY=<openai_api_key>
    ```
