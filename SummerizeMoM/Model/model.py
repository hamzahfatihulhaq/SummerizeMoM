from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import os

def VicunaModels(model_name, config):
    os.environ['CURL_CA_BUNDLE'] = ''
    
    model_name_or_path = model_name

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.get("max_new_tokens"),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        repetition_penalty=config.get("repetition_penalty")
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# ValueError: Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object