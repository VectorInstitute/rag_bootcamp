import random
from pprint import pprint
import sys

from llama_index import ServiceContext, set_global_service_context, set_global_handler, SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter

from task_dataset import PubMedQATaskDataset

sys.path.append("..")
from utils.hosting_utils import RAGLLM
from utils.rag_utils import (
    DocumentReader, RAGEmbedding, RAGQueryEngine, extract_yes_no, evaluate, validate_rag_cfg
)
from utils.storage_utils import RAGIndex


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
        "retriever_type": "vector_index", # "vector_index", "bm25"
        "retriever_similarity_top_k": 3,
        "query_mode": "hybrid", # "default", "hybrid"
        "hybrid_search_alpha": 0.5, # float from 0.0 (sparse search - bm25) to 1.0 (vector search)
        "response_mode": "compact",
    }

    ### STAGE 0 - Preliminary config checks
    pprint(rag_cfg)
    validate_rag_cfg(rag_cfg)

    ### STAGE 1 - Load dataset and documents
    # 1. Load dataset from local folder
    print('Loading PubMed QA data ...')
    reader = DocumentReader(input_dir="./data/pubmed_doc")
    docs = reader.load_data()
    print(f'No. of documents loaded: {len(docs)}')

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
    index = RAGIndex(db_type=rag_cfg['vector_db_type'], db_name=rag_cfg['vector_db_name']).create_index(docs)
    
    ### STAGE 4 - Build query engine
    query_engine = RAGQueryEngine(
    retriever_type=rag_cfg['retriever_type'],DocumentReader, RAGEmbedding, RAGQueryEngine, extract_yes_no, evaluate, validate_rag_cfg vector_index=index, llm_model_name=rag_cfg['llm_name']).create(
        similarity_top_k=rag_cfg['retriever_similarity_top_k'], response_mode=rag_cfg['response_mode'], 
        query_mode=rag_cfg["query_mode"], hybrid_search_alpha=rag_cfg["hybrid_search_alpha"])
    
    ### STAGE 5 - Finally query the model
    random.seed(237)
    sample_idx = random.randint(0, len(docs)-1)
    sample_elm = docs[sample_idx]
    pprint(sample_elm)


if __name__ == "__main__":

    main()
