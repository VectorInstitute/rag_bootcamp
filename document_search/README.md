# RAG Bootcamp: Document Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place in February-March 2024. Popular LLMs like Cohere and OpenAI are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. Use a collection of unstructured documents to answer domain-specific questions, like: *"How many AI scholarships did Vector Institute award in 2022?"*

In this demo, we augment a Cohere LLM request with an unstructured PDF document search to get the correct answer: **Vector Institute awarded 109 AI scholarships in 2022**

## Requirements

* Python 3.10+
* Cohere API key saved in your home directory at `~/.cohere.key`