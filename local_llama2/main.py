import random
from pprint import pprint
import sys

from llama_index import ServiceContext, set_global_service_context, set_global_handler, SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter

sys.path.append("..")
from utils.hosting_utils import RAGLLM
from utils.rag_utils import (
    DocumentReader, RAGEmbedding, RAGQueryEngine, validate_rag_cfg
)


def main():

    # Set RAG configuration
    rag_cfg = {
        # Node parser config
        "chunk_size": 256,
        "chunk_overlap": 0,

        # Embedding model config
        "embed_model_type": "hf",
        "embed_model_name": "BAAI/bge-base-en-v1.5",

        # LLM config
        "llm_type": "local",
        "llm_name": "Llama-2-7b-chat-hf",
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "do_sample": False,

        # Vector DB config
        "vector_db_type": "chromadb",
        "vector_db_name": "local_llama2",

        # Retriever and query config
        "retriever_type": "bm25", # "vector_index", "bm25"
        "retriever_similarity_top_k": 3,
        "query_mode": "hybrid", # "default", "hybrid"
        "hybrid_search_alpha": 0.5, # float from 0.0 (sparse search - bm25) to 1.0 (vector search)
        "response_mode": "compact",
    }

    ### STAGE 0 - Preliminary config checks
    pprint(rag_cfg)
    validate_rag_cfg(rag_cfg)

    ### STAGE 1 - Load dataset and documents
    # 1. Load PubMed QA dataset
    print('Loading documents...')
    pdf_folder_path = "./source-materials"
    print(f"*** Loading source materials from {pdf_folder_path}\n")
    docs = SimpleDirectoryReader(pdf_folder_path).load_data()
    print(f"Loaded data size: {len(docs)}")

    # 2. Load documents
    print('Loading documents ...')
    pdf_folder_path = "./source-materials"
    print(f"*** Loading source materials from {pdf_folder_path}\n")
    docs = SimpleDirectoryReader(pdf_folder_path).load_data()

    ### STAGE 2 - Load node parser, embedding, LLM and set service context
    # 1. Load node parser to split documents into smaller chunks
    print('Loading node parser ...')
    node_parser = SentenceSplitter(chunk_size=rag_cfg['chunk_size'], chunk_overlap=rag_cfg['chunk_overlap'])
    # nodes = node_parser.get_nodes_from_documents(docs)

    # 2. Load embedding model
    embed_model = RAGEmbedding(model_type=rag_cfg['embed_model_type'], model_name=rag_cfg['embed_model_name']).load_model()

    # 3. Load LLM for generation
    llm = RAGLLM(rag_cfg['llm_type'], rag_cfg['llm_name']).load_model(**rag_cfg) # TODO - pass args

    # 4. Use service context to set the node parser, embedding model, LLM, etc.
    # TODO - Add pompt_helper (if required)
    service_context = ServiceContext.from_defaults(
        node_parser=node_parser,
        embed_model=embed_model,
        llm=llm,
    )
    # Set it globally to avoid passing it to every class, this sets it even for rag_utils.py
    set_global_service_context(service_context)


    ### STAGE 3 - Create index using the appropriate vector store
    """
    index = RAGIndex(db_type=rag_cfg['vector_db_type'], db_name=rag_cfg['vector_db_name'])\
        .create_index(docs, weaviate_url=rag_cfg["weaviate_url"])
    

    ### STAGE 4 - Build query engine
    # Now build a query engine using retriever, response_synthesizer and node_postprocessor (add this later)
    query_engine_args = {
        "similarity_top_k": rag_cfg['retriever_similarity_top_k'], 
        "response_mode": rag_cfg['response_mode'],
    }
    if (rag_cfg["retriever_type"] == "vector_index") and (rag_cfg["vector_db_type"] == "weaviate"):
        query_engine_args.update({
            "query_mode": rag_cfg["query_mode"], 
            "hybrid_search_alpha": rag_cfg["hybrid_search_alpha"]
        })
    elif rag_cfg["retriever_type"] == "bm25":
        nodes = service_context.node_parser.get_nodes_from_documents(docs)
        tokenizer = service_context.embed_model._tokenizer
        query_engine_args.update({"nodes": nodes, "tokenizer": tokenizer})
    query_engine = RAGQueryEngine(
        retriever_type=rag_cfg['retriever_type'], vector_index=index, llm_model_name=rag_cfg['llm_name']).create(**query_engine_args)

    result_dict = evaluate(pubmed_data, query_engine)
    output_dict = {
        "num_samples": len(pubmed_data),
        "config": rag_cfg,
        "result": result_dict,
    }
    pprint(output_dict)
    """


if __name__ == "__main__":

    main()
