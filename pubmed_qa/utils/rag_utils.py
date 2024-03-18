import os
import re
import numpy as np

from tqdm import tqdm
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, 
    load_index_from_storage, get_response_synthesizer, download_loader,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank, SentenceEmbeddingOptimizer
from llama_index.postprocessor.cohere_rerank import CohereRerank
from langchain_community.chat_models import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from datasets import Dataset
from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision
        )
from ragas import evaluate as ragas_evaluate


RAGAS_METRIC_MAP = {
        "faithfulness": faithfulness,
        "relevancy": answer_relevancy,
        "recall": context_recall,
        "precision": context_precision
        }


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
        elif self._file_ext == '.jsonl':
            JSONReader = download_loader("JSONReader")
            reader = JSONReader()
            docs = []
            for file in os.listdir(self.input_dir):
                docs.extend(reader.load_data(os.path.join(self.input_dir, file), is_jsonl=True))
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
    LlamaIndex supports embedding models from OpenAI, Cohere, HuggingFace, etc.
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    We can also build out custom embedding model: 
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#custom-embedding-model
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
        self.node_postprocessor = None
        self.response_synthesizer = None

    def create(self, similarity_top_k, response_mode, **kwargs):
        self.set_retriever(similarity_top_k, **kwargs)
        self.set_response_synthesizer(response_mode)
        if kwargs["use_reranker"]:
            self.set_node_postprocessors(rerank_top_k=kwargs["rerank_top_k"])
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=self.node_postprocessor,
            response_synthesizer=self.response_synthesizer,
            )
        return query_engine
    
    def set_retriever(self, similarity_top_k, **kwargs):
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
                vector_store_query_mode=kwargs["query_mode"],
                alpha=kwargs["hybrid_search_alpha"],
                )
        elif self.retriever_type == 'bm25':
            self.retriever = BM25Retriever(
                nodes=kwargs["nodes"],
                tokenizer=kwargs["tokenizer"],
                similarity_top_k=similarity_top_k,
            )
        else:
            raise NotImplementedError(f'Incorrect retriever type - {self.retriever_type}')

    def set_node_postprocessors(self, rerank_top_k=2):
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
        cohere_rerank = CohereRerank(top_n=rerank_top_k)
        self.node_postprocessor = [cohere_rerank]

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

def retriever_acc(actual, retrieved_candidates):
    # TODO - Implement other metrics like recall@k
    # candidates are unordered as of now
    return (actual in retrieved_candidates)

def evaluate(data, engine):
    gt_ans = []
    pred_ans = []
    retriever_hit = []
    for elm in tqdm(data, desc="Running evaluation"):
        query_str = elm['question']
        resp = engine.query(query_str)
        ans = extract_yes_no(resp.response).lower()
        gt_ans.append(elm['answer'][0])
        pred_ans.append(ans)
        
        # Standalone retriever accuracy
        try:
            ret_nodes = engine.retriever.retrieve(query_str)
            retriever_hit.append(retriever_acc(
                elm['id'], [node.metadata["file_name"].split(".")[0] for node in ret_nodes]))
        except Exception as e:
            print(f"Exception for {elm['id']}: {e}")
            retriever_hit.append(0)

    acc = [(gt_ans[idx]==pred_ans[idx]) for idx in range(len(gt_ans))]
    return {"acc": np.mean(acc), "retriever_acc": np.mean(retriever_hit)}

def validate_rag_cfg(cfg):
    if cfg["query_mode"] == "hybrid":
        assert cfg["hybrid_search_alpha"] is not None, "hybrid_search_alpha cannot be None if query_mode is set to 'hybrid'"
    if cfg["vector_db_type"] == "weaviate":
        assert cfg["weaviate_url"] is not None, "weaviate_url cannot be None for weaviate vector db"


class RagasEval():
    def __init__(self, metrics, eval_llm_type, eval_llm_name):
        self.eval_llm_type = eval_llm_type # "openai", "cohere", "local"
        self.eval_llm_name = eval_llm_name # "gpt-3.5-turbo" # "gpt-4"

        self.temperature = 0.0

        # self.local_embed_name = "BAAI/bge-base-en-v1.5"
        # self.local_model_path = "/home/omkar/model-weights"
        # self.local_llm_name = "Llama-2-7b-chat-hf"

        self._prepare_embedding()
        self._prepare_llm()

        self.metrics = [RAGAS_METRIC_MAP[elm] for elm in metrics]

    def _prepare_data(self, data):
        return Dataset.from_dict(data)

    def _prepare_embedding(self):
        if self.eval_llm_type == "cohere":
            self.eval_embedding = CohereEmbeddings(
                    model="embed-english-v3.0"
                    )
        elif self.eval_llm_type == "local":
            self.eval_embedding = HuggingFaceBgeEmbeddings(
                    model_name=self.local_embed_name,
                    )
        elif self.eval_llm_type == "openai":
            self.eval_embedding = None
    
    def _prepare_llm(self):
        if self.eval_llm_type == "cohere":
            self.eval_llm = ChatCohere(
                    model="command",
                    )
        elif self.eval_llm_type == "local":
            self.eval_llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-2-7b-chat-hf",
                    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                    )
        elif self.eval_llm_type == "openai":
            self.eval_llm = ChatOpenAI(
                model_name=self.eval_llm_name,
                temperature=self.temperature,
                )

    def evaluate(self, data):
        data = self._prepare_data(data)
        result = ragas_evaluate(
                    data,
                    metrics=self.metrics,
                    embeddings=self.eval_embedding,
                    llm=self.eval_llm,
                )
        return result