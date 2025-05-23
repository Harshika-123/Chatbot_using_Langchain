from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

#llm=HuggingFaceEndpoint(
#    repo_id='google/gemma-2-2b-it',
#    task='text-generation'
#)
#model=ChatHuggingFace(llm=llm)
model=ChatOpenAI()

# !st prompt -> detailed report
template1 =PromptTemplate(
    template="write a detailed report on{topic}",
    input_variables=['topic']
    )

# 2 prompt
template2 =PromptTemplate(
    template="write a 5 line summary on the following text. /n report on{text}",
    input_variables=['text']
    )

parser =StrOutputParser()

chain=template1|model|parser|template2|model|parser

result= chain.invoke({'topic':'black hole'})

print(result)
