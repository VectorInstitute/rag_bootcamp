# RAG Bootcamp: Cloud Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place in February-March 2024. Popular LLMs like Cohere and OpenAI are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we use a data loader from [Llama Hub](https://llamahub.ai/) to access an AWS S3 bucket and search documents stored in the cloud.

## Requirements

* Python 3.10+
* Cohere API key saved in your home directory at `~/.cohere.key`
* AWS API key saved in your home directory at `~/.aws/credentials` in the following format:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
