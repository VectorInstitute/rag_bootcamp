# RAG Bootcamp

This is a collection of reference implementations for Vector Institute's RAG (Retrieval-Augmented Generation) bootcamp, scheduled to take place in 2024.

## Reference Implementations

### MedGPT

Look under the `/medgpt` folder. This reference implementation requires an OpenAI API key.

## Python environment

We are trying to make a single Python virtual environment that will work for all the reference implementations included here. Start by building the environment:

```
python3 -m venv --python="/pkgs/python-3.8/bin/python3" "/ssd003/projects/aieng/public/rag_bootcamp/rag_venv"
cd /ssd003/projects/aieng/public/rag_bootcamp/rag_venv
source bin/activate
python3 -m pip install -r requirements.txt
```

To activate the environment and start using it:

```
cd /ssd003/projects/aieng/public/rag_bootcamp
source rag_venv/bin/activate
```
