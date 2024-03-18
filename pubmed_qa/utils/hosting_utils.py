from llama_index.llms.huggingface import HuggingFaceLLM

class RAGLLM():
    '''
    LlamaIndex supports OpenAI, Cohere, AI21 and HuggingFace LLMs
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    '''
    def __init__(self, llm_type, llm_name):
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.local_model_path = "/model-weights"

    def load_model(self, **kwargs):
        print(f'Loading {self.llm_type} LLM model ...')
        gen_arg_keys = ['temperature', 'top_p', 'top_k', 'do_sample']
        gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_arg_keys}
        if self.llm_type == 'local':
            # Using local HuggingFace LLM stored at /model-weights
            llm = HuggingFaceLLM(
                tokenizer_name=f"{self.local_model_path}/{self.llm_name}",
                model_name=f"{self.local_model_path}/{self.llm_name}",
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
