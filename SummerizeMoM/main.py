from striprtf.striprtf import rtf_to_text
from gensim.summarization.summarizer import summarize
from Model.model import VicunaModels
from Summarization.map_reduce_sum import MapReduceSum
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import os 
import yaml
from time import time

def dir_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Create MoM with LLM Model")
    parser.add_argument("--file", type=dir_path, help="file .RTF path to create the MoM file", required=True )
    parser.add_argument("--summary", type=str, help="Name file for summary", const="hasil_summary.txt", nargs='?', required=False )
    parser.add_argument("--MoM", type=str, help="Name file for MoM", const="MoM_Summery.txt", nargs='?', required=False )
    args = parser.parse_args()
    
    return args

def textSplitter(essay, chunkSize, chunkOverlap):
    print(f"chunkSize : {chunkSize}")
    text_splitter = RecursiveCharacterTextSplitter(separators=[".", "\n\n", "\n"], chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    docs = text_splitter.create_documents([essay])

    for i in range(0,len(docs)):
        print(llm.get_num_tokens(docs[i].page_content))
    
    return docs

if __name__=="__main__":
    # GET ARGUMENTS
    args = parse_args()

    # Generate MODEL
    with open("SummerizeMoM/config.yaml","r") as file_object:
        data=yaml.load(file_object,Loader=yaml.SafeLoader)

    model_name= data["model"]
    config= data["config"]

    llm = VicunaModels(model_name, config)

    # GET FILES
    path = args.file

    start = time()

    with open(path) as infile:
        content = infile.read()

    text = rtf_to_text(content)

    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>')

    summ = summarize(text)

    # write summary
    if args.summary ==  None:
        args.summary  = "hasil_summary.txt"

    with open(args.summary, "w") as output_file:
       output_file.write(summ)
    
    # Summerize into max requirement
    essay = summ
    maxTokenGenerate = 9000
    while(llm.get_num_tokens(essay) >  maxTokenGenerate):
        essay = summarize(essay, word_count= 3000)

    chunkSize = 3000
    chunkOverlap = 0
    docs = textSplitter(essay, chunkSize, chunkOverlap)


    # MOM
    result=MapReduceSum(llm,docs)
    if args.MoM ==  None:
        args.MoM  = "MoM_Summery.txt"
    with open(args.MoM, "w") as output_file:
        output_file.write(result)

    print(f"time process : {start - time()} s")
    