from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import Literal,TypedDict

class review_state(TypedDict):
    review:str
    sentiment:Literal['positive','negative']
    diagnosis:dict
    response:str

model=ChatGroq(
    model="openai/gpt-oss-20b",
    
)

class sentimentschema(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="sentiment of the review")

structured_model=model.with_structured_output(sentimentschema)

def find_sentiment(state:review_state):
    prompt=f"for the following review give the sentiment \n {state['review']}"
    res=structured_model.invoke(prompt)
    return {'sentiment':res.sentiment}


def check_sentiment(state:review_state):
    if state['sentiment']=='positive':
        return "positive_response"
    else:
        return "diagnosis_review"

def positive_response(state:review_state):
    prompt=f"""write a warm thank you message in response  to this review \n {state['review']} also kindly ask the user th give the feed back on our website"""
    res=model.invoke(prompt)
    return {'response':res.content}

class DiagnosisSchema(BaseModel):
    issue_type:Literal["UX","performence","bug","support","other"]=Field(description="the category of isuue metion in the review")
    tone:Literal["angry","frustrated","disappointed","calm"]=Field(description="the emotional tone expressed by the user")
    urgency:Literal["low",'medium','high']=Field(description="how argent or critical the issue appear to be")

diagnosis_structured_model=model.with_structured_output(DiagnosisSchema)
def diagnosis_review(state:review_state):
    prompt=f"""diagnosis this negative review \n {state['review']} \n return issuetype,tone and urgency"""
    res=diagnosis_structured_model.invoke(prompt)
    return {'diagnosis':res.model_dump()}

def negative_response(state:review_state):
    diagnosis=state['diagnosis']
    prompt=f"""your support assistent the user had a{diagnosis['issue_type']} ,sounded {diagnosis['tone']},and marked argency as {diagnosis['urgency']} write a empathetic helpful resolution message"""
    res=model.invoke(prompt)
    return {'response':res.content}

graph=StateGraph(review_state)
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("positive_response",positive_response)
graph.add_node("negative_response",negative_response)
graph.add_node("diagnosis_review",diagnosis_review)

graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("positive_response",END)
graph.add_edge("diagnosis_review","negative_response")
graph.add_edge("negative_response",END)

workflow=graph.compile()
result=workflow.invoke({'review':"the product is so bad"})
print(result['response'])

