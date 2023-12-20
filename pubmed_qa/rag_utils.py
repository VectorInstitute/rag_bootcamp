import os
import re
import numpy as np
import chromadb

from tqdm import tqdm
from llama_index import (
    SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, 
    load_index_from_storage, get_response_synthesizer
)
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.llms import HuggingFaceLLM, OpenAI
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor, LLMRerank, SentenceEmbeddingOptimizer


class DocumentReader():

    def __init__(self, input_dir, exclude_llm_metadata_keys=True, exclude_embed_metadata_keys=True):
        self.input_dir = input_dir
        self._file_ext = os.path.splitext(os.listdir(input_dir)[0])[1]

        self.exclude_llm_metadata_keys = exclude_llm_metadata_keys
        self.exclude_embed_metadata_keys = exclude_embed_metadata_keys

    def load_data(self):
        docs = None
        # Use reader based on file extension of documents
        # Only support '.txt' files as of now
        if self._file_ext == '.txt':
            reader = SimpleDirectoryReader(input_dir=self.input_dir)
            docs = reader.load_data()
        else:
            raise NotImplementedError(f'Does not support {self._file_ext} file extension for document files.')
        
        # Can choose if metadata need to be included as input when passing the doc to LLM or embeddings: 
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html
        # Exclude metadata keys from embeddings or LLMs based on flag
        if docs is not None:
            all_metadata_keys = list(docs[0].metadata.keys())
            if self.exclude_llm_metadata_keys:
                for doc in docs:
                    doc.excluded_llm_metadata_keys = all_metadata_keys
            if self.exclude_embed_metadata_keys:
                for doc in docs:
                    doc.excluded_embed_metadata_keys = all_metadata_keys

        return docs


class RAGEmbedding():
    '''
    Llama-index supports embedding models from OpenAI, Cohere, LangChain, HuggingFace, etc. 
    We can also build out custom embedding model. 
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    '''
    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

    def load_model(self):
        print(f'Loading {self.model_type} embedding model ...')
        if self.model_type == 'hf':
            # Using bge base HuggingFace embeddings, can choose others based on leaderboard: 
            # https://huggingface.co/spaces/mteb/leaderboard
            embed_model = HuggingFaceEmbedding(model_name=self.model_name) # max_length does not have any effect?

        elif self.model_type == 'openai':
            # TODO - Add open ai embedding model
            # embed_model = OpenAIEmbedding()
            raise NotImplementedError

        # sample_text_embedding = embed_model.get_text_embedding(sample_text)
        # print(sample_text_embedding)
        # print(sample_text)
        # print(len(sample_text_embedding))
        
        return embed_model


class RAGLLM():
    '''
    Llama-index supports OpenAI, Cohere, AI21 and HuggingFace LLMs
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    '''
    def __init__(self, llm_type, llm_name):
        self.llm_type = llm_type
        self.llm_name = llm_name

    def load_model(self, **kwargs):
        print(f'Loading {self.llm_type} LLM model ...')
        gen_arg_keys = ['temperature', 'top_p', 'top_k', 'do_sample']
        gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_arg_keys}
        if self.llm_type == 'local':
            # Using local HuggingFace LLM stored at /model-weights
            llm = HuggingFaceLLM(
                tokenizer_name=f"/model-weights/{self.llm_name}",
                model_name=f"/model-weights/{self.llm_name}",
                device_map="auto",
                context_window=4096,
                max_new_tokens=kwargs['max_new_tokens'],
                generate_kwargs=gen_kwargs,
                # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
            )
        elif self.llm_type == 'openai':
            # TODO - Add open ai llm
            # llm = OpenAI()
            raise NotImplementedError

        return llm


class RAGIndex():
    '''
    Use storage context to set custom vector store
    Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    '''
    def __init__(self, db_type, db_name):
        self.db_type = db_type
        self.db_name = db_name
        self._persist_dir = './.index_store/'

    def create_index(self, docs, save=True):
        # Only supports ChromaDB as of now
        if self.db_type == 'chromadb':
            chroma_client = chromadb.Client()
            chroma_collection = chroma_client.create_collection(name=self.db_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        else:
            raise NotImplementedError(f'Incorrect vector db type - {self.db_type}')
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        if save:
            if not os.path.isdir(self._persist_dir):
                os.mkdir(self._persist_dir)
            index.storage_context.persist(persist_dir=self._persist_dir)
            # TODO - Figure out reload

        return index


class RAGQueryEngine():
    '''
    https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    TODO - Check other args for RetrieverQueryEngine
    '''
    def __init__(self, retriever_type, vector_index, llm_model_name):
        self.retriever_type = retriever_type
        self.index = vector_index
        self.llm_model_name = llm_model_name
        self.retriever = None
        self.node_postprocessors = None
        self.response_synthesizer = None

    def create(self, similarity_top_k, response_mode):
        self.set_retriever(similarity_top_k)
        self.set_response_synthesizer(response_mode)
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            # node_postprocessors=self.node_postprocessor
            response_synthesizer=self.response_synthesizer,
            )
        return query_engine
    
    def set_retriever(self, similarity_top_k):
        # Other retrievers can be used based on the type of index: List, Tree, Knowledge Graph, etc.
        # https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers.html
        # Find LlamaIndex equivalents for the following:
        # Check MultiQueryRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
        # Check Contextual compression from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
        # Check Ensemble Retriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
        # Check self-query from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
        # Check WebSearchRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
        if self.retriever_type == 'vector_index':
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                )
        else:
            raise NotImplementedError(f'Incorrect retriever type - {self.retriever_type}')

    def set_node_postprocessors(self):
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
        raise NotImplementedError

    def set_response_synthesizer(self, response_mode):
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
        if self.llm_model_name == 'Llama-2-7b-chat-hf':
            llama2_chat_tmpl = ("<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{qa_prompt} [/INST]")
            llama2_chat_sys_msg = (
                "You are a helpful and honest assistant. "
                "Your answer should only be based on the context information provided."
            )
            qa_prompt_tmpl = llama2_chat_tmpl.format_map({'sys_msg': llama2_chat_sys_msg, 'qa_prompt': qa_prompt_tmpl})
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)

        self.response_synthesizer = get_response_synthesizer(response_mode=response_mode, text_qa_template=qa_prompt_tmpl)


def extract_yes_no(resp):
    match_pat = '[^\w](yes|no)[^\w]'
    match_txt = re.search(match_pat, resp, re.IGNORECASE)
    if match_txt:
        match_txt = match_txt.group(0)
    else:
        return 'none'
    clean_txt = re.sub('[^\w]', '', match_txt)
    return clean_txt

def evaluate(data, engine):
    gt_ans = []
    pred_ans = []
    for elm in tqdm(data):
        resp = engine.query(elm['question'])
        ans = extract_yes_no(resp.response).lower()
        gt_ans.append(elm['answer'][0])
        pred_ans.append(ans)
    acc = [(gt_ans[idx]==pred_ans[idx]) for idx in range(len(gt_ans))]
    return np.mean(acc)