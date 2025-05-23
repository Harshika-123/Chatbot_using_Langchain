#from langchain_openai import ChatOpenAI
#from dotenv import load_dotenv
#from langchain_core.prompts import PromptTemplate 

#load_dotenv()

##llm=HuggingFaceEndpoint(
##    repo_id='google/gemma-2-2b-it',
##    task='text-generation'
##)
##model=ChatHuggingFace(llm=llm)
#model=ChatOpenAI()

## !st prompt -> detailed report
#template1 =PromptTemplate(
#    template="write a detailed report on{topic}",
#    input_variables=['topic']
#    )

## 2 prompt
#template2 =PromptTemplate(
#    template="write a 5 line summary on the following text. /n report on{text}",
#    input_variables=['text']
#    )
#prompt1=template1.invoke({'topic':'black hole'})

#result= model.invoke(prompt1)

#prompt2 = template2.invoke({'text':result.content})

#result1=model.invoke(prompt2)
#print(result1.content)
##----------------------------

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Use GPT-2 model for text-generation
pipe = pipeline(
    "text-generation",
    model="gpt2",          # <-- Changed here to a free, stable open source model
    max_length=256,
    temperature=0.7,
    top_p=0.95,
)

llm = HuggingFacePipeline(pipeline=pipe)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',  # fixed newline here
    input_variables=['text']
)

# Generate detailed report prompt
prompt1 = template1.invoke({'topic':'black hole'})

# Get detailed report result from model
result = llm.invoke(prompt1)

# Generate summary prompt based on detailed report text
prompt2 = template2.invoke({'text': result})


# Get summary result from model
result1 = llm.invoke(prompt2)

print(result1)