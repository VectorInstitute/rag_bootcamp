#!/usr/bin/env python3

from getpass import getpass
import os
from pathlib import Path

from llama_index import download_loader, ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import Cohere
from llama_index.postprocessor.cohere_rerank import CohereRerank


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():

    # Setup the environment
    try:
        os.environ["COHERE_API_KEY"] = open(Path.home() / ".cohere.key", "r").read().strip()
        os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]
    except Exception:
        print(f"Unable to read your Cohere API key. Make sure this is stored in a text file in your home directory at ~/.cohere.key")

    # Start with making a generation request without RAG augmentation
    query = "What are the risk factors of Tesla for the year 2023?"
    llm = Cohere(api_key=os.environ["COHERE_API_KEY"])
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    result = llm.complete(query)
    print(f"Result: {result}\n")

    # Now use the SECFilingsLoader module to get some specific data
    SECFilingsLoader = download_loader('SECFilingsLoader')
    loader = SECFilingsLoader(tickers=['TSLA'],amount=3,filing_type="10-K")
    loader.load_data()
    documents = SimpleDirectoryReader("data\TSLA\2022").load_data()
    #index = VectorStoreIndex.from_documents(documents)
    print(f"Number of source materials: {len(documents)}\n")
    print(f"Example first source material:\n {documents[0]}\n")
    
    # Define Embeddings Model
    print(f"*** Setting up the embeddings model...")
    embed_model = CohereEmbedding(
        model_name="embed-english-v3.0",
        input_type="search_query"
    )
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm
    )

    # Set up the base vector store retriever
    print(f"*** Setting up the base vector store retriever\n")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    search_query_retriever = index.as_retriever(service_context=service_context)
 
    # Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
    search_query_retriever = index.as_retriever(service_context=service_context)
    search_query_retrieved_nodes = search_query_retriever.retrieve(query)
    print(f"Search query retriever found {len(search_query_retrieved_nodes)} results")
    print(f"First result example:\n{search_query_retrieved_nodes[0]}\n")

    # Apply reranking with CohereRerank
    print(f"*** Applying re-ranking with CohereRerank, and then sending the original query again\n")
    reranker = CohereRerank()
    query_engine = index.as_query_engine(
        node_postprocessors = [reranker]
    )

    # Now ask the original query again, this time augmented with reranked results
    result = query_engine.query(query)
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()

