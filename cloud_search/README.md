# RAG Bootcamp: Cloud Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jab 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we use a data loader from [Llama Hub](https://llamahub.ai/) to access an AWS S3 bucket and search documents stored in the cloud.

## Requirements

* Python 3.10+
* AWS API key saved in your home directory at `~/.aws/credentials` in the following format:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
