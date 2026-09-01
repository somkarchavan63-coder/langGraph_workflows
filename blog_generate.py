"""from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace"""
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_groq import ChatGroq
import os

from langchain_core.output_parsers import StrOutputParser

class blog_state(TypedDict):
    topic:str
    out_line:str
    fin_blog:str

"""llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    provider="auto",
    task="text-generation",
    max_new_tokens=600,
    temperature=0.7,
    
    
)"""
chat_model=ChatGroq(
    model="openai/gpt-oss-20b",

)

"""chat_model=ChatHuggingFace(llm=llm)"""


#function to create the outline
def blog_outline(state:blog_state):
    topic=state["topic"]

    prompt=f"give the outline for this topic {topic}"

    response=chat_model.invoke(prompt)

    state['out_line']=response.content
    return state

#function to generate the blog on given topic and outline

def generate_blog(state:blog_state):
    parser=StrOutputParser()
    outline=state["out_line"]

    prompt=f"generate the blog for this outline {outline}"

    response=chat_model.invoke(prompt)
    result=parser.invoke(response)
    
    state["fin_blog"]=result

    return state


graph=StateGraph(blog_state)

graph.add_node("blog_outline",blog_outline)
graph.add_node("generate_blog",generate_blog)

graph.add_edge(START,"blog_outline")
graph.add_edge("blog_outline","generate_blog")
graph.add_edge("generate_blog",END)

workflow=graph.compile()

blog=workflow.invoke({'topic':"Esports gaming"})
print(blog['out_line'])

