from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain

# Map
template = """
Tulislah summery dalam Bahasa Indonesia dari kutipan berikut ini:
"{text}"
ringkasan:
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

combine_prompt = """
Buat 'Minutes of Meeting' dalam Bahasa Indonesia dengan isi
A. Judul Rapat
B. Outline
C. Key Point
D. Rekomendasi/Action Plan
dari Resume berikut:
"
{text}
"
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

def MapReduceSum(llm, summary):
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=combine_prompt_template)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=7000,
    )

    map_chain = LLMChain(llm=llm, prompt=prompt)

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="text",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    result=map_reduce_chain.run(summary)
    
    return result