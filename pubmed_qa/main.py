from pathlib import Path
from pprint import pprint
import subprocess
import sys
import os
import random

from llama_index.core import ServiceContext, set_global_service_context, set_global_handler
from llama_index.core.node_parser import SentenceSplitter

from task_dataset import PubMedQATaskDataset

sys.path.append("..")
from utils.hosting_utils import RAGLLM
from utils.rag_utils import (
    DocumentReader, RAGEmbedding, RAGQueryEngine, RagasEval, 
    extract_yes_no, evaluate, validate_rag_cfg
    )
from utils.storage_utils import RAGIndex


with open(Path.home() / ".cohere_api_key", "r") as f:
    os.environ["COHERE_API_KEY"] = f.read().rstrip("\n")
# with open(Path.home() / ".hfhub_api_token", "r") as f:
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = f.read().rstrip("\n")
with open(Path.home() / ".openai_api_key", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")


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
        "vector_db_type": "weaviate", # "chromadb", "weaviate"
        "vector_db_name": "Pubmed_QA",
        # MODIFY THIS
        "weaviate_url": "https://rag-bootcamp-pubmed-qa-lsqv7od4.weaviate.network",

        # Retriever and query config
        "retriever_type": "bm25", # "vector_index", "bm25"
        "retriever_similarity_top_k": 3,
        "query_mode": "hybrid", # "default", "hybrid"
        "hybrid_search_alpha": 0.5, # float from 0.0 (sparse search - bm25) to 1.0 (vector search)
        "response_mode": "compact",
        "use_reranker": True,
        "rerank_top_k": 2,

        # Evaluation config
        "eval_llm_type": "openai",
        "eval_llm_name": "gpt-3.5-turbo",
    }

    # # Set handler for debugging
    # # https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html
    # set_global_handler("simple")

    # Setup the environment
    try:
        f = open(Path.home() / ".weaviate_api_key", "r")
        f.close()
    except Exception as err:
        print(f"Could not read your Weaviate key. Please make sure this is available in plain text under your home directory in ~/.weaviate_api_key: {err}")

    ### STAGE 0 - Preliminary config checks
    pprint(rag_cfg)
    validate_rag_cfg(rag_cfg)


    ### STAGE 1 - Load dataset and documents
    # 1. Load PubMed QA dataset
    print('Loading PubMed QA data ...')
    pubmed_data = PubMedQATaskDataset('bigbio/pubmed_qa')
    print(f"Loaded data size: {len(pubmed_data)}")
    pubmed_data.mock_knowledge_base(output_dir='./data', one_file_per_sample=True)

    # 2. Load documents
    print('Loading documents ...')
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
    if rag_cfg["use_reranker"]:
        query_engine_args.update({"use_reranker": True, "rerank_top_k": rag_cfg["rerank_top_k"]})
    query_engine = RAGQueryEngine(
        retriever_type=rag_cfg['retriever_type'], vector_index=index, llm_model_name=rag_cfg['llm_name']).create(**query_engine_args)


    ### STAGE 5 - Finally query the model!
    random.seed(41)
    sample_idx = random.randint(0, len(pubmed_data)-1)
    sample_elm = pubmed_data[sample_idx]
    # print(sample_elm)

    query = sample_elm['question']
    response = query_engine.query(query)
    print(f'QUERY: {query}')
    print(f'RESPONSE: {response}')
    print(f'YES/NO: {extract_yes_no(response.response)}')
    print(f'GT ANSWER: {sample_elm["answer"]}')
    print(f'GT LONG ANSWER: {sample_elm["long_answer"]}')

    retrieved_nodes = query_engine.retriever.retrieve(query)
    # print(f"GT doc ID: {sample_elm['id']}")
    # print(query)
    # for node in retrieved_nodes:
    #     print(node.metadata["file_name"].split(".")[0])
    #     print(node.text)
    #     print(node.score)
    #     print('\n')
    
    ## Ragas evaluation
    eval_data = {
        "question": [query],
        "answer": [response.response],
        "contexts": [[node.text for node in retrieved_nodes]],
        "ground_truths": [[sample_elm['long_answer']]],
        }
    print(eval_data)
    eval_obj = RagasEval(
        metrics=["faithfulness", "relevancy", "recall", "precision"], 
        eval_llm_type=rag_cfg["eval_llm_type"], eval_llm_name=rag_cfg["eval_llm_name"]
        )
    eval_result = eval_obj.evaluate(eval_data)
    print(eval_result)

#    result_dict = evaluate(pubmed_data, query_engine)
#    output_dict = {
#        "num_samples": len(pubmed_data),
#        "config": rag_cfg,
#        "result": result_dict,
#    }
#    pprint(output_dict)

    # print(f'Overall Acc: {evaluate(pubmed_data, query_engine)}') 
    # # Chroma DB: 500 samples - Overall Acc: 0.626
    # # Weaviate DB: 500 samples -
    # # alpha 0.5 - 0.596
    # # alpha 1.0 - 0.61
    # # alpha 0.0 - 0.562

    # {'config': {'chunk_overlap': 0,
    #         'chunk_size': 256,
    #         'do_sample': False,
    #         'embed_model_name': 'BAAI/bge-base-en-v1.5',
    #         'embed_model_type': 'hf',
    #         'hybrid_search_alpha': 1.0,
    #         'llm_name': 'Llama-2-7b-chat-hf',
    #         'llm_type': 'local',
    #         'max_new_tokens': 256,
    #         'query_mode': 'hybrid',
    #         'response_mode': 'compact',
    #         'retriever_similarity_top_k': 3,
    #         'retriever_type': 'vector_index',
    #         'temperature': 1.0,
    #         'top_k': 50,
    #         'top_p': 1.0,
    #         'vector_db_name': 'Pubmed_QA',
    #         'vector_db_type': 'weaviate',
    #         'weaviate_url': 'https://vector-rag-lab-xsxuylwh.weaviate.network'},
    # 'num_samples': 500,
    # 'result': {'acc': 0.666, 'retriever_acc': 0.994}}
    # Chunk ablation --------------
    # 'result': {'acc': 0.666, 'retriever_acc': 0.994}} for {'chunk_overlap': 32, 'chunk_size': 128,}
    # 'result': {'acc': 0.666, 'retriever_acc': 0.992}} for {'chunk_overlap': 0, 'chunk_size': 64,}


if __name__ == "__main__":

    main()
