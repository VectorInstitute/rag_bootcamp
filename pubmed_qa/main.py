import os
import argparse
import random
import chromadb
import re

from pathlib import Path
import llama_index
from llama_index import (
    VectorStoreIndex, ServiceContext, PromptTemplate,
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
from rag_utils import DocumentReader, extract_yes_no, evaluate


def main():

    # Set RAG configuration, TODO - Add elements while developing
    embed_model_type = 'hf' # Type of embedding model to choose from
    llm_type = 'local' # Whether to load local LLM or use Open AI API 
    
    print('Loading PubMed QA data ...')
    pubmed_data = PubMedQATaskDataset('bigbio/pubmed_qa')
    print(len(pubmed_data))
    # print(pubmed_data[9])
    # pubmed_data.mock_knowledge_base(output_dir='./data')

    # # Set handler for debugging
    # # https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html
    # llama_index.set_global_handler("simple")
    

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
    # 1. Load embedding model
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
            tokenizer_name="/model-weights/Llama-2-7b-chat-hf",
            model_name="/model-weights/Llama-2-7b-chat-hf",
            device_map="auto",
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 1.0, "top_p": 1.0, "do_sample": False}, # greedy decoding
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

    # 2. Create/load index using the appropriate vector store
    # Use storage context to set custom vector store
    # Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    # Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    # LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(name="pubmed_qa")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    
    # # Persist index to disk and reload
    # persist_dir="./index_store/"
    # index.storage_context.persist(persist_dir=persist_dir)
    # storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # index = load_index_from_storage(storage_context)


    ### QUERYING STAGE
    # Now build a query engine using retriever, response_synthesizer and node_postprocessor (add this later)
    # https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    # TODO - Check other args for RetrieverQueryEngine

    # Other retrievers can be used based on the type of index: List, Tree, Knowledge Graph, etc.
    # https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers.html
    # Find LlamaIndex equivalents for the following:
    # Check MultiQueryRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
    # Check Contextual compression from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
    # Check Ensemble Retriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
    # Check self-query from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
    # Check WebSearchRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        )

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

    # Other response modes: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#configuring-the-response-mode
    qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. "
    "If your answer is in favour of the query, end your response with 'yes' otherwise end your response with 'no'.\n"
    "Query: {query_str}\n"
    "Answer: "
    )
    llama2_chat_tmpl = ("<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{qa_prompt} [/INST]")
    llama2_chat_sys_msg = (
        "You are a helpful and honest assistant. "
        "Your answer should only be based on the context information provided."
    )
    qa_prompt_tmpl = llama2_chat_tmpl.format_map({'sys_msg': llama2_chat_sys_msg, 'qa_prompt': qa_prompt_tmpl})
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)
    response_synthesizer = get_response_synthesizer(response_mode='compact', text_qa_template=qa_prompt_tmpl) # compact

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        # node_postprocessors=node_postprocessor
        response_synthesizer=response_synthesizer,
    )

    # # Finally query the model!
    # random.seed(41)
    # sample_idx = random.randint(0, len(pubmed_data)-1)
    # sample_elm = pubmed_data[sample_idx]
    # # print(sample_elm)

    # query = sample_elm['question']
    # print(f'QUERY: {query}\n')

    # retrieved_nodes = retriever.retrieve(query)
    # for node in retrieved_nodes:
    #     print(node.text)
    #     print(node.score)
    #     print('\n')

    # response = query_engine.query(query)

    # print(f'QUERY: {query}')
    # print(f'RESPONSE: {response}')
    # print(f'YES/NO: {extract_yes_no(response.response)}')
    # print(f'GT ANSWER: {sample_elm["answer"]}')
    # print(f'GT LONG ANSWER: {sample_elm["long_answer"]}')

    # pubmed_data = pubmed_data[:5]

    print(f'Overall Acc: {evaluate(pubmed_data, query_engine)}')


if __name__ == "__main__":

    # # Set OpenAI API key
    # os.environ["OPENAI_API_KEY"] = Path("./.openai_key").read_text()

    main()