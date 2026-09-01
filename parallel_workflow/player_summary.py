from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser

class player_state(TypedDict):
    runs:int
    bolls:int
    fours:int
    sixes:int
    sr:float
    boundry_per:float
    bpb:float
    summary:str


llm=ChatGroq(
    model="openai/gpt-oss-20b",
    
)
#calculating the strike rate
def strike_rate(state:player_state):
    runs=state['runs']
    bolls=state['bolls']
    st_rate=(runs/bolls)*100
    
    return {'sr':st_rate}


#calculating the boundry percentage

def boundry_per(state:player_state):
    fours=state['fours']
    sixes=state['sixes']
    runs=state['runs']
    total_boun=(fours*4)+(sixes*6)
    b_per=(total_boun/runs)*100
    
    return {'boundry_per':b_per}

#function for calculating the boundries per boll
def bolles_p_bound(state:player_state):
    fours=state['fours']
    sixes=state['sixes']
    bolls=state['bolls']
    tot_boun=fours+sixes
    bpb=(tot_boun/bolls)*100
    
    return {'bpb':bpb}

#generating the summary using llm
def summary(state:player_state):
    parser=StrOutputParser()
    runs=state['runs']
    bolls=state['bolls']
    fours=state['fours']
    sixes=state['sixes']
    str_rate=state['sr']
    bound_per=state['boundry_per']
    bpb=state['bpb']

    prompt=f"generate the summary of the player based on his performence runs {runs},bolls {bolls},fours {fours},sixes {sixes},strike rate {str_rate},boundry percentage {bound_per} and bolls per boundries {bpb}"
    response=llm.invoke(prompt)
    result=parser.invoke(response)
    state['summary']=result
    return state

graph=StateGraph(player_state)

graph.add_node("strike_rate",strike_rate)
graph.add_node("boundry_per",boundry_per)
graph.add_node("bolles_p_bound",bolles_p_bound)
graph.add_node("summary",summary)

graph.add_edge(START,"strike_rate")
graph.add_edge(START,"boundry_per")
graph.add_edge(START,"bolles_p_bound")

graph.add_edge("strike_rate","summary")
graph.add_edge("boundry_per","summary")
graph.add_edge("bolles_p_bound",'summary')

graph.add_edge("summary",END)

workflow=graph.compile()
result=workflow.invoke({'runs':100,'bolls':50,'fours':3,'sixes':2})
print(result['summary'])