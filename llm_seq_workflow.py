from langgraph.graph import StateGraph,START,END
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from typing import TypedDict

class llmstate(TypedDict):
    question:str
    answer:str

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    provider="auto",
    task="text-generation",
    max_new_tokens=360,
    temperature=0.7,
    
)
chat_model=ChatHuggingFace(llm=llm)

def chat_llm(state:llmstate):
    question=state["question"]

    prompt=f"answer the following question {question}"

    answer=chat_model.invoke(prompt)

    state["answer"]=answer

    return state

graph=StateGraph(llmstate)

graph.add_node("chat_llm",chat_llm)

graph.add_edge(START,"chat_llm")
graph.add_edge("chat_llm",END)

workflow=graph.compile()

initial_state=({'question':"what is the capital city of INDIA"})

final_state=workflow.invoke(initial_state)

print(final_state['answer'].content)