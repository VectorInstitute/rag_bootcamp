import faiss
import os
import weaviate

from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from .rag_utils import get_embed_model_dim


class RAGIndex:
    """
    Use storage context to set custom vector store
    Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    """

    def __init__(self, db_type, db_name):
        self.db_type = db_type
        self.db_name = db_name
        self._persist_dir = f"./.{db_type}_index_store/"

    def create_index(self, docs, save=True, **kwargs):
        # Only supports Weaviate as of now
        if self.db_type == "weaviate":
            with open(Path.home() / ".weaviate.key", "r") as f:
                weaviate_api_key = f.read().rstrip("\n")
            weaviate_client = weaviate.connect_to_wcs(
                cluster_url=kwargs["weaviate_url"],
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            )
            vector_store = WeaviateVectorStore(
                weaviate_client=weaviate_client,
                index_name=self.db_name,
            )
        elif self.db_type == "local":
            # Use FAISS vector database for local index
            faiss_dim = get_embed_model_dim(kwargs["embed_model"])
            faiss_index = faiss.IndexFlatL2(faiss_dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
        else:
            raise NotImplementedError(f"Incorrect vector db type - {self.db_type}")

        if os.path.isdir(self._persist_dir):
            # Load if index already saved
            print(f"Loading index from {self._persist_dir} ...")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=self._persist_dir,
            )
            index = load_index_from_storage(storage_context)
        else:
            # Re-index
            print("Creating new index ...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                docs, storage_context=storage_context
            )
            if save:
                os.makedirs(self._persist_dir, exist_ok=True)
                index.storage_context.persist(persist_dir=self._persist_dir)

        return index
