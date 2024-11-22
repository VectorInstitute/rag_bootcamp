# RAG Bootcamp: Cloud Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we use a data loader from [Llama Hub](https://llamahub.ai/) to access an AWS S3 bucket and search documents stored in the cloud.

## Requirements

1. Python 3.10+
2. If running on the Vector cluster, load the ```rag_dataloaders``` env:
    ```bash
    source /ssd003/projects/aieng/public/rag_bootcamp/envs/rag_dataloaders/bin/activate
    ```
   Otherwise, create your own env and install packages using the ```requirements.txt``` file:
    ```bash
    python3 -m venv ./rag_dataloaders
    source rag_dataloaders/bin/activate
    python3 -m pip install -r envs/rag_dataloaders/requirements.txt
    ```
3. Install the jupyter kernel:
    ```bash
    ipython kernel install --user --name=rag_dataloaders
    ```
4. If you are accessing the LLMs and embedding models through Vector AI Engineering's Kaleidoscope Service (Vector Inference + Autoscaling), you will need to set the KScope env variables:
    - Request a KScope API Key:
        Run the following command (replace ```<user_id>``` and ```<password>```) from **within the cluster** to obtain the API Key. The ```access_token``` in the output is your KScope API Key.
        ```bash
        curl -X POST -d "grant_type=password" -d "username=<user_id>" -d "password=<password>" https://kscope.vectorinstitute.ai/token
        ```
    - After obtaining the `.env` configurations, make sure to create the ```.kscope.env``` file in your home directory (```/h/<user_id>```) and set the following env variables:
        ```bash
        export OPENAI_BASE_URL="https://kscope.vectorinstitute.ai/v1"
        export OPENAI_API_KEY=<kscope_api_key>
        ```
   If you using OpenAI models instead, set the env variables to the following values:
        ```bash
        export OPENAI_BASE_URL="https://api.openai.com/v1"
        export OPENAI_API_KEY=<openai_api_key>
        ```
5. AWS API key saved in your home directory at `~/.aws/credentials` in the following format:
    ```bash
    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY
    ```
