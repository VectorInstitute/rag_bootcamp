#!/usr/bin/env python3

# Following code sample modified from https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c

from getpass import getpass
import os
from pathlib import Path

from langchain.chat_models import ChatCohere
from langchain.schema import HumanMessage
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.text_splitter import SentenceSplitter


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
    if not os.path.exists("source-materials"):
        os.mkdir("source-materials")

    # Start with making a generation request without RAG augmentation
    query = "What is Vector Institute doing to address AI safety and trustworthiness?"
    llm = ChatCohere()
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    message = [
        HumanMessage(
            content=query
        )
    ]
    result = llm(message)
    print(f"Result: {result.content}\n")

    # Load the pdf
    pdf_folder_path = "./source-materials"
    print(f"*** Loading source materials from {pdf_folder_path}\n")
    documents = SimpleDirectoryReader(pdf_folder_path).load_data()
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
    cohere_rerank = CohereRerank(top_n=3)
    query_engine = index.as_query_engine(
        node_postprocessors = [cohere_rerank]
    )

    # Now ask the original query again, this time augmented with reranked results
    result = query_engine.query(query)
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()

