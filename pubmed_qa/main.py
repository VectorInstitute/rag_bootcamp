import os
import argparse
# import chromadb

from pathlib import Path
import llama_index
from llama_index import (
    VectorStoreIndex, ServiceContext, 
    set_global_service_context, load_index_from_storage, get_response_synthesizer,
)
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.llms import HuggingFaceLLM, OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor, LLMRerank, SentenceEmbeddingOptimizer
from llama_index.schema import MetadataMode

from task_dataset import PubMedQATaskDataset
from rag_utils import DocumentReader


def main():

    # Set RAG configuration, TODO - Add elements while developing
    embed_model_type = 'hf' # Type of embedding model to choose from
    llm_type = 'local' # Whether to load local LLM or use Open AI API 
    
    print('Loading PubMed QA data ...')
    pubmed_data = PubMedQATaskDataset('bigbio/pubmed_qa')
    print(len(pubmed_data))
    # print(pubmed_data[9])
    # pubmed_data.mock_knowledge_base(output_dir='./data')

    # Set handler for debugging
    # https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html
    llama_index.set_global_handler("simple")
    
    ### LOADING STAGE
    # 1. Load data
    print('Loading documents ...')
    reader = DocumentReader(input_dir="./data/pubmed_doc")
    docs = reader.load_data()
    print(f'No. of documents loaded: {len(docs)}')

    # 2. Split documents into smaller chunks
    print('Loading node parser ...')
    chunk_size = 256
    chunk_overlap = 0
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # nodes = node_parser.get_nodes_from_documents(docs)
    # print(len(nodes))

    ### INDEXING STAGE
    # Load embedding model
    # Llama-index supports embedding models from OpenAI, Cohere, LangChain, HuggingFace, etc. 
    # We can also build out custom embedding model.
    # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    print(f'Loading {embed_model_type} embedding model ...')
    if embed_model_type == 'hf':
        # Using bge base HuggingFace embeddings, can choose others based on leaderboard: 
        # https://huggingface.co/spaces/mteb/leaderboard
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5") # max_length does not have any effect?
    elif embed_model_type == 'openai':
        # TODO - Add open ai embedding model
        # embed_model = OpenAIEmbedding()
        raise NotImplementedError
    # sample_text = nodes[0].text
    # sample_text_embedding = embed_model.get_text_embedding(sample_text)
    # print(sample_text_embedding)
    # print(sample_text)
    # print(len(sample_text_embedding))

    # Choose and set LLM for generation
    # Llama-index supports OpenAI, Cohere, AI21 and HuggingFace LLMs
    # https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    # Using local HuggingFace LLM - Llama-2-7b
    print(f'Loading {llm_type} LLM model ...')
    if llm_type == 'local':
        llm = HuggingFaceLLM(
            tokenizer_name="/model-weights/Llama-2-7b-hf",
            model_name="/model-weights/Llama-2-7b-hf",
            device_map="auto",
            context_window=4096,
            max_new_tokens=128,
            generate_kwargs={"temperature": 0.8},
            # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
        )
    elif llm_type == 'openai':
        # TODO - Add open ai llm
        # llm = OpenAI()
        raise NotImplementedError

    # Use service context to set the LLM, embedding model, etc.
    # Set it globally to avoid passing it to every class
    # TODO - Add pompt_helper (if required)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        node_parser=node_parser,
        llm=llm,
    )
    set_global_service_context(service_context)

    # # Use storage context to set custom vector store
    # # Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    # # Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    # # LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    # chroma_client = chromadb.EphemeralClient()
    # chroma_collection = chroma_client.create_collection("quickstart")
    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # TODO - Fix package issue with Chroma

    # # Create the index
    # index = VectorStoreIndex.from_documents(docs) # storage_context=storage_context
    
    # # # Persist index to disk and reload
    # # persist_dir="./index_store/"
    # # index.storage_context.persist(persist_dir=persist_dir)
    # # storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # # index = load_index_from_storage(storage_context)


    # ### QUERYING STAGE
    # # Now build a query engine using retriever, response_synthesizer and node_postprocessor (add this later)
    # # https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    # # TODO - Check other args for RetrieverQueryEngine

    # # Other retrievers can be used based on the type of index: List, Tree, Knowledge Graph, etc.
    # # https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers.html
    # # Find LlamaIndex equivalents for the following:
    # # Check MultiQueryRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
    # # Check Contextual compression from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
    # # Check Ensemble Retriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
    # # Check self-query from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
    # # Check WebSearchRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
    # retriever = VectorIndexRetriever(
    #     index=index,
    #     similarity_top_k=5,
    #     )

    # # Node postprocessor: Porcessing nodes after retrieval before passing to the LLM for generation
    # # Re-ranking step can be performed here!
    # # Nodes can be re-ordered to include more relevant ones at the top: https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder
    # # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html
    # # node_postprocessor = [SimilarityPostprocessor(similarity_cutoff=0.5)]
    # # node_postprocessor = [LLMRerank(top_n=2)]
    # node_postprocessor = [
    #     SentenceEmbeddingOptimizer(
    #         embed_model=service_context.embed_model, 
    #         percentile_cutoff=0.5
    #         )]

    # response_synthesizer = get_response_synthesizer()

    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     node_postprocessors=node_postprocessor
    # )

    # # Finally query the model!
    # query = "Describe the model editing method used in the paper?"
    # response = query_engine.query(query)

    # print(f"{''.join(['*']*50)}")
    # print(f"chunk args: {(chunk_size, chunk_overlap)}\n")
    # print(f"response: {response}\n")



if __name__ == "__main__":

    # # Set OpenAI API key
    # os.environ["OPENAI_API_KEY"] = Path("./.openai_key").read_text()

    main()