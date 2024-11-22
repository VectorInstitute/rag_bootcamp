# RAG Bootcamp: Evaluation Techniques

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on.

When implementing **RAG (Retrieval-Augmented Generation)** workflows, it is very difficult to understand if you are getting high quality results. There are many different techniques to evaluate RAG workflows, as described in many online sources such as https://weaviate.io/blog/rag-evaluation. In these reference implementations, we focus on the [Ragas](https://docs.ragas.io/en/stable/) framework.

There are many different metrics to determine if a RAG workflow is producing high "quality" results. The Ragas framework defines these metrics at https://docs.ragas.io/en/latest/concepts/metrics/index.html. In these examples, we focus on the following three:

- **Faithfulness:** The generated answer is regarded as *faithful* if all the claims that are made in the answer can be inferred from the given context(s).
- **Answer Correctness:** Was the generated answer correct? Was it complete?
- **Context Precision:** Did our retriever return good results that matched the question it was being asked?

## Requirements

1. Python 3.10+
2. If running on the Vector cluster, load the `rag_evaluation` env:
    ```bash
    source /ssd003/projects/aieng/public/rag_bootcamp/envs/rag_evaluation/bin/activate
    ```
   **[Alternate]** Otherwise, create your own env and install packages using the `requirements.txt` file (make sure to run these commands in the top level dir of the repository):
    ```bash
    python3 -m venv ./rag_evaluation
    source rag_evaluation/bin/activate
    python3 -m pip install -r ./envs/rag_evaluation/requirements.txt
    ```
3. Install the jupyter kernel:
    ```bash
    ipython kernel install --user --name=rag_evaluation
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
