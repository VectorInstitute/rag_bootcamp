# RAG Bootcamp: Local Llama2
Most of the reference implementations in this guide rely on external services like Cohere and Weaviate for various parts of the RAG workflows. In some cases this might not be desirable, such as when dealing with sensitive data. This **Local Llama2** reference implementation shows how to run a RAG workflow entirely in a local environment, using open source tooling, and not depending on any third-party services.


## Requirements
* Python 3.10
* A system with an NVIDIA GPU and CUDA drivers installed. The GPU needs to have at least 48 GB of video memory. (We are working to make the model offloadable to cpu system memory, hoping to lower this requirement)


## Installation Steps

### Clone the Github repo:

    git clone git@github.com:/VectorInstitute/rag_bootcamp

### Start by installing the `rag_local` Python environment:

    cd ~/rag_bootcamp/envs
    python3 -m venv rag_local
    source rag_local/bin/activate
    python3 -m pip install -r rag_local/requirements.txt

### Download the Llama2-7B weights:

There are many ways to do this, but the easiest is to download directly from Huggingface. Put these under a folder called `/model-weights`

    mkdir /model-weights
    cd /model-weights
    git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Lastly, start a Jupyter notebook and open the `document_search_local_llama2.ipynb` notebook:

    cd ~/rag_bootcamp
    jupyter notebook --ip $(hostname --fqdn)

