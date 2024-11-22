# RAG Bootcamp: Evaluation Techniques

When implementing **RAG (Retrieval-Augmented Generation)** workflows, it is very difficult to understand if you are getting high quality results.

There are many different techniques to evaluate RAG workflows, as described in many online sources such as https://weaviate.io/blog/rag-evaluation. In these reference implementations, we focus on the [Ragas](https://docs.ragas.io/en/stable/) framework.

There are also many different metrics to determine if a RAG workflow is producing high "quality" results. The Ragas framework defines many different such metrics at https://docs.ragas.io/en/latest/concepts/metrics/index.html. In these examples, we focus on the following three:

- **Faithfulness:** The generated answer is regarded as *faithful* if all the claims that are made in the answer can be inferred from the given context(s).
- **Answer Correctness:** Was the generated answer correct? Was it complete?
- **Context Precision:** Did our retriever return good results that matched the question it was being asked?

## Requirements

* Python 3.10+
* OpenAI API key
