from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import operator 

generate_model=ChatGroq(
    model="openai/gpt-oss-20b",
    
)

eval_model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    
    temperature=0.7
)

optimize_model=ChatGroq(
    model="openai/gpt-oss-20b",
    
)

class tweetstate(TypedDict):
    topic:str
    tweet:str
    evalution:Literal["approved","need_improvement"]
    feedback:str
    iteration:int
    max_iteration:int
    tweet_history:Annotated[list[str],operator.add]
    feedback_history:Annotated[list[str],operator.add]

class tweetevalution(BaseModel):
    evaluation:Literal["approved","need_improvement"]=Field(description="evaluate the tweet")
    feedback:str=Field(description="give the feedback to improve")

structured_model=eval_model.with_structured_output(tweetevalution,method="json_schema")

def generate_tweet(state:tweetstate):
    messages = [
    SystemMessage(content="You are a funny and clever Twitter/X influencer."),
    HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english.
""")
]
    res=generate_model.invoke(messages)
    return {'tweet':res.content,'feedback_history':[res.content]}

def eval_tweet(state:tweetstate):
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality - Is this fresh, or have you seen it a hundred times before?
2. Humor - Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness - Is it short, sharp, and scroll-stopping?
4. Virality Potential - Would people retweet or share it?
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Don't end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
]
    res=structured_model.invoke(messages)
    return {"evalution":res.evaluation,"feedback":res.feedback,'tweet_history':[res.feedback]}

def optimize_tweet(state:tweetstate):
    messages = [
    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
    HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
]
    res=optimize_model.invoke(messages)
    iteration=state['iteration']+1
    return {'tweet':res.content,'iteration':iteration,'tweet_history':[res.content]}

def route_eval(state:tweetstate):
    if state['evalution']=="approved" :
        return "approved"
    if state["iteration"]>=state['max_iteration']:
        return "approved"
    
    return "optimize_tweet"

graph=StateGraph(tweetstate)
graph.add_node("generate_tweet",generate_tweet)
graph.add_node("eval_tweet",eval_tweet)
graph.add_node("optimize_tweet",optimize_tweet)

graph.add_edge(START,"generate_tweet")
graph.add_edge("generate_tweet","eval_tweet")
graph.add_conditional_edges("eval_tweet",route_eval,{'approved':END,'optimize_tweet':"optimize_tweet"})
graph.add_edge('optimize_tweet',"eval_tweet")

workflow=graph.compile()
initial_state={
    'topic':'India railway',
    'iteration':1,
    'max_iteration':5,
}
res=workflow.invoke(initial_state)
print(res)
print(res['tweet'])
print(res['iteration'])
print(res['max_iteration'])
print(res['tweet_history'])
print(res['feedback_history'])
print(res['feedback'])
