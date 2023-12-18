import os
from llama_index import SimpleDirectoryReader


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
