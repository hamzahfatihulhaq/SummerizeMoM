from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import os

def VicunaModels(model_name, config):
    os.environ['CURL_CA_BUNDLE'] = ''
    
    model_name_or_path = model_name

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="balanced", revision="main")
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