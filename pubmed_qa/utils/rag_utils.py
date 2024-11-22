import os
import re

# from pathlib import Path

from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from llama_index.core import (
    SimpleDirectoryReader,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader
from llama_index.retrievers.bm25 import BM25Retriever

from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    ResponseRelevancy,
)


RAGAS_METRIC_MAP = {
    "faithfulness": Faithfulness(),
    "relevancy": ResponseRelevancy(),
    "recall": NonLLMContextRecall(),
    "precision": NonLLMContextPrecisionWithReference(),
}


class DocumentReader:
    def __init__(
        self,
        input_dir,
        exclude_llm_metadata_keys=True,
        exclude_embed_metadata_keys=True,
    ):
        self.input_dir = input_dir
        self._file_ext = os.path.splitext(os.listdir(input_dir)[0])[1]

        self.exclude_llm_metadata_keys = exclude_llm_metadata_keys
        self.exclude_embed_metadata_keys = exclude_embed_metadata_keys

    def load_data(self):
        docs = None
        # Use reader based on file extension of documents
        # Only support '.txt' files as of now
        if self._file_ext == ".txt":
            reader = SimpleDirectoryReader(input_dir=self.input_dir)
            docs = reader.load_data()
        elif self._file_ext == ".jsonl":
            reader = JSONReader()
            docs = []
            for file in os.listdir(self.input_dir):
                docs.extend(
                    reader.load_data(os.path.join(self.input_dir, file), is_jsonl=True)
                )
        else:
            raise NotImplementedError(
                f"Does not support {self._file_ext} file extension for document files."
            )

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


class RAGEmbedding:
    """
    LlamaIndex supports embedding models from OpenAI, Cohere, HuggingFace, etc.
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    We can also build out custom embedding model:
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#custom-embedding-model
    """

    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

    def load_model(self):
        print(f"Loading {self.model_type} embedding model ...")
        if self.model_type == "hf":
            # Using bge base HuggingFace embeddings, can choose others based on leaderboard:
            # https://huggingface.co/spaces/mteb/leaderboard
            embed_model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device="cuda",
                trust_remote_code=True,
            )  # max_length does not have any effect?

        elif self.model_type == "openai":
            # TODO - Add OpenAI embedding model
            # embed_model = OpenAIEmbedding()
            raise NotImplementedError

        return embed_model


class RAGQueryEngine:
    """
    https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    TODO - Check other args for RetrieverQueryEngine
    """

    def __init__(self, retriever_type, vector_index):
        self.retriever_type = retriever_type
        self.index = vector_index
        self.retriever = None
        self.node_postprocessor = None
        self.response_synthesizer = None

    def create(self, similarity_top_k, response_mode, **kwargs):
        self.set_retriever(similarity_top_k, **kwargs)
        self.set_response_synthesizer(response_mode=response_mode)
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
        if self.retriever_type == "vector_index":
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=kwargs["query_mode"],
                alpha=kwargs["hybrid_search_alpha"],
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                nodes=kwargs["nodes"],
                tokenizer=kwargs["tokenizer"],
                similarity_top_k=similarity_top_k,
            )
        else:
            raise NotImplementedError(
                f"Incorrect retriever type - {self.retriever_type}"
            )

    def set_node_postprocessors(self, rerank_top_k=2):
        # Node postprocessor: Porcessing nodes after retrieval before passing to the LLM for generation
        # Re-ranking step can be performed here!
        # Nodes can be re-ordered to include more relevant ones at the top: https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html

        self.node_postprocessor = [LLMRerank(top_n=rerank_top_k)]

    def set_response_synthesizer(self, response_mode):
        # Other response modes: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#configuring-the-response-mode
        qa_prompt_tmpl = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query while providing an explanation. "
            "If your answer is in favour of the query, end your response with 'yes' otherwise end your response with 'no'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)

        self.response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_prompt_tmpl,
            response_mode=response_mode,
        )


class RagasEval:
    def __init__(
        self, metrics, eval_llm_type, eval_llm_name, embed_model_name, **kwargs
    ):
        self.eval_llm_type = eval_llm_type  # "openai", "cohere", "local", "kscope"
        self.eval_llm_name = eval_llm_name

        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 256)

        self.embed_model_name = embed_model_name

        self._prepare_embedding()
        self._prepare_llm()

        self.metrics = [RAGAS_METRIC_MAP[elm] for elm in metrics]

    def _prepare_data(self, data):
        return EvaluationDataset.from_list(data)

    def _prepare_embedding(self):
        model_kwargs = {"device": "cuda", "trust_remote_code": True}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity

        self.eval_embedding = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=self.embed_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        )

    def _prepare_llm(self):
        if self.eval_llm_type == "local":
            self.eval_llm = LangchainLLMWrapper(
                HuggingFaceEndpoint(
                    repo_id=f"meta-llama/{self.eval_llm_name}",
                    temperautre=self.temperature,
                    max_new_tokens=self.max_tokens,
                    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                )
            )
        elif self.eval_llm_type == "kscope":
            self.eval_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=self.eval_llm_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
        elif self.eval_llm_type == "openai":
            self.eval_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=self.eval_llm_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=os.environ["RAGAS_OPENAI_BASE_URL"],
                    api_key=os.environ["RAGAS_OPENAI_API_KEY"],
                )
            )
        elif self.eval_llm_type == "cohere":
            self.eval_llm = LangchainLLMWrapper(
                ChatCohere(
                    model=self.eval_llm_name,
                )
            )

    def evaluate(self, data):
        eval_data = self._prepare_data(data)
        result = ragas_evaluate(
            dataset=eval_data,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embedding,
        )
        return result


def extract_yes_no(resp):
    match_pat = r"\b(?:yes|no)\b"
    match_txt = re.search(match_pat, resp, re.IGNORECASE)
    if match_txt:
        return match_txt.group(0)
    return "none"


def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)


def validate_rag_cfg(cfg):
    if cfg["query_mode"] == "hybrid":
        assert (
            cfg["hybrid_search_alpha"] is not None
        ), "hybrid_search_alpha cannot be None if query_mode is set to 'hybrid'"
    if cfg["vector_db_type"] == "weaviate":
        assert (
            cfg["weaviate_url"] is not None
        ), "weaviate_url cannot be None for weaviate vector db"
