from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated
import operator

class eval_state(TypedDict):
    essay:str
    clarity_feedback:str
    analysis_feedback:str
    lang_feedback:str
    individual_score:Annotated[list[float],operator.add]
    summary:str
    avg_score:float

model=ChatGroq(
    model="openai/gpt-oss-20b",
    
)

class evaluationschema(BaseModel):
    feedback:str=Field(description="detailed feedback for the essay")
    score:float=Field(description="score out of 10",ge=0,le=10)

structured_model=model.with_structured_output(evaluationschema)
essay="""# Artificial Intelligence

Artificial Intelligence (AI) is one of the most transformative technologies of the modern era. It refers to the ability of machines and computer systems to perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, understanding language, recognizing images, and making decisions. AI has moved from being a concept in science fiction to becoming an important part of everyday life.

The development of AI has been gradual. Early AI systems were designed to follow predefined rules and solve specific problems. With the advancement of computing power, large datasets, and machine learning algorithms, AI systems became capable of learning patterns from data rather than relying only on manually written rules. Today, technologies such as machine learning, deep learning, natural language processing, and computer vision form major areas of AI.

AI is already being used in many industries. In healthcare, it can assist doctors in analyzing medical images, identifying patterns, and supporting diagnosis. In education, AI-powered systems can provide personalized learning experiences and help students understand difficult concepts. In banking and finance, AI is used for fraud detection, risk analysis, and customer service. In transportation, AI contributes to navigation systems, traffic prediction, and autonomous vehicle research. Businesses also use AI-powered chatbots and recommendation systems to improve customer experiences.

One of the most visible developments in recent years is generative AI. Unlike traditional systems that mainly classify or predict information, generative AI can create new content such as text, images, audio, video, and computer code. Large language models are an example of generative AI that can understand and produce human-like text. These systems are increasingly being used for writing, programming, research, education, and creative work.

Despite its benefits, AI also creates several challenges. AI systems can sometimes produce incorrect or biased information because they learn from data that may contain errors or biases. Privacy is another concern because many AI applications depend on large amounts of data. There are also concerns about job displacement as automation becomes more capable of performing certain tasks. Therefore, responsible development, transparency, human oversight, and appropriate regulations are important for ensuring that AI is used safely and fairly.

The future of AI is likely to involve even greater integration with human activities. AI may help scientists discover new medicines, improve climate forecasting, develop more efficient technologies, and solve complex problems. However, AI should be viewed as a tool that complements human intelligence rather than simply replacing it. Human creativity, judgment, empathy, and ethical reasoning remain extremely important.

In conclusion, Artificial Intelligence has the potential to significantly improve the way people live and work. Its applications are expanding rapidly across healthcare, education, business, science, and many other fields. At the same time, its risks and limitations must be carefully addressed. The future impact of AI will depend not only on how powerful the technology becomes, but also on how responsibly humans develop and use it.

"""    

"""prompt=f"evaluate the language quality of essay and provide feedback and assign score out of 10 \n {essay}"
print(structured_model.invoke(prompt))"""

#function for clearity feedback and score

def clarity_feedback(state:eval_state):
    prompt=f"evaluate the clarity of thought of essay and provide feedback and assign score out of 10 \n {state['essay']}"
    res=structured_model.invoke(prompt)
    return {'clarity_feedback':res.feedback,'individual_score':[res.score]}

def analysis_feedback(state:eval_state):
    prompt=f"evaluate the deapth of analysis of essay and provide the feedback and assign score out of 10 \n{state['essay']}"
    res=structured_model.invoke(prompt)
    return {"analysis_feedback":res.feedback,"individual_score":[res.score]}

def lang_feedback(state:eval_state):
    prompt=f"evaluate the language quality of the essay and provide the feedback and assign score out of 10 \n {state['essay']}"
    res=structured_model.invoke(prompt)
    return {'lang_feedback':res.feedback,'individual_score':[res.score]}

def summary(state:eval_state):
    prompt=f"based on following feedback create the summarize feedback \n clarity of thoughts feedback-{state['clarity_feedback']} \n deapth of analysis feedback-{state['analysis_feedback']} \n language quality feedback-{state['lang_feedback']}"
    overall_feedback=model.invoke(prompt)
    avg_score=sum(state['individual_score'])/len(state['individual_score'])

    return {"summary":overall_feedback.content,"avg_score":avg_score}


graph=StateGraph(eval_state)
graph.add_node("clarity_feedback",clarity_feedback)
graph.add_node("analysis_feedback",analysis_feedback)
graph.add_node("lang_feedback",lang_feedback)
graph.add_node("summary",summary)


graph.add_edge(START,"clarity_feedback")
graph.add_edge(START,"analysis_feedback")
graph.add_edge(START,"lang_feedback")

graph.add_edge("clarity_feedback","summary")
graph.add_edge("analysis_feedback","summary")
graph.add_edge("lang_feedback","summary")

graph.add_edge("summary",END)

workflow=graph.compile()
result=workflow.invoke({'essay':essay})
print(result['essay'])
print(result['avg_score'])


print(result['clarity_feedback'])