import sys
import io
import os
from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import re
from IPython.display import display, Image, Markdown
from langchain_core.runnables.graph import MermaidDrawMethod
import operator
from transformers import pipeline
# Set API key
os.environ["OPENAI_API_KEY"] = ""

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
# Initialize Ollama LLM
#llm = ChatOllama(model="mistral", temperature=0, base_url="http://localhost:11434")  # Change "mistral" to your desired Ollama model
#llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

class State(TypedDict):
    """State of the graph."""

    query: str
    code: str
    debugged_code: str
    explanation: str
    test_cases: str
    execution_result: str
    documentation: str
    next: Literal[
        "code_generator_agent",
        "debugger_agent",
        "code_explainer_agent",
        "test_case_generator",
        "execution_agent",
        "documentation_agent",
    ]
    
# Define agent functions
def user_query_processor(state: State):
    """Routes user queries to the appropriate agent."""
    print("--User Query Agent--")
    query = state["query"]
    
    if "debug" in query.lower():
        return {"next": "code_generator_agent"} # changed from "debugger_agent" to "code_generator_agent"
    elif "explain" in query.lower():
        return {"next": "code_explainer_agent"}
    elif "test case" in query.lower():
        return {"next": "test_case_generator"}
    elif "run" in query.lower():
        return {"next": "execution_agent"}
    elif "document" in query.lower():
        return {"next": "documentation_agent"}
    else:
        return {"next": "code_generator_agent"}

def code_generator(state: State):
    """Generates code based on user prompt."""
    print("--Code Generator Agent--")
    prompt = f"Write a python code for: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    #print(response)
    return {"code": response.content}

def debugger_agent(state: State):
    """Finds and fixes errors in code."""
    print("--Debugger Agent--")
    code = state["code"]
    prompt = f"Debug this code:\n\n{code}\n\nProvide a corrected version."
    response = llm.invoke([HumanMessage(content=prompt)])
    #print(response.content)
    return {"debugged_code": response.content}

def code_explainer_agent(state: State):
    """Explains the code logic."""
    print("--Code Explainer Agent--")
    code = state["query"]
    prompt = f"Explain the following code in simple terms:\n\n{code}"
    response = llm.invoke([HumanMessage(content=prompt)])
   # print(display(Markdown(response.content)))
    return {"code": response.content}

def test_case_generator(state: State):
    """Generates test cases for the given code."""
    print("--Test Case Generator Agent--")
    query = state["query"]
    prompt = f"Generate test cases for the following code:\n\n{query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"code": response.content}


def execution_agent(state: dict):
    """Executes the Python code and returns the output or errors."""
    print("--Execution Agent--")
    query = state.get("query", "")

    # Extract actual Python code from the query (assuming structured input)
    if "Run the following code" in query:
        query = query.split("Run the following code", 1)[-1].strip()

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        exec_globals = {}
        exec(query, exec_globals)
        output = stdout_capture.getvalue().strip()
        return {"code": output if output else "Code executed successfully."}
    except Exception as e:
        return {"code": f"Error: {str(e)}"}
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def documentation_agent(state: State):
    """Generates documentation for the code."""
    print("--Documentation Agent--")
    code = state["code"]
    prompt = f"Generate Markdown documentation for the following code/note:\n\n{code}"
    response = llm.invoke([HumanMessage(content=prompt)])
    print(display(Markdown(response.content)))
    return response

# Define Graph
workflow = StateGraph(State)

# Define the router node
def router(state):
    return state["next"]

workflow.add_node("router", user_query_processor)

# Add nodes (Agents)
workflow.add_node("code_generator_agent", code_generator)
workflow.add_node("debugger_agent", debugger_agent)
workflow.add_node("code_explainer_agent", code_explainer_agent)
workflow.add_node("test_case_generator", test_case_generator)
workflow.add_node("execution_agent", execution_agent)
workflow.add_node("documentation_agent", documentation_agent)

# Define edges (connections)
workflow.set_entry_point("router")

# add a conditional edge to decide if we should test_case_generator or documentation_agent
def decide_to_continue(state):
    """Check if the user wants to continue."""
    return "documentation_agent"

workflow.add_conditional_edges(
    "code_generator_agent",
    decide_to_continue,
    {
        "test_case_generator": "test_case_generator",
        "documentation_agent": "documentation_agent",
    },
)

# add a direct edge from code_generator to debugger
workflow.add_edge("code_generator_agent", "debugger_agent")
workflow.add_edge("code_explainer_agent", "documentation_agent")
workflow.add_edge("test_case_generator", "documentation_agent")
workflow.add_edge("execution_agent", "documentation_agent")
# add a conditional edge to route from router to any of our other nodes.
workflow.add_conditional_edges(
    "router",
    router,
    {
        "code_generator_agent": "code_generator_agent",
        "debugger_agent": "debugger_agent",
        "code_explainer_agent": "code_explainer_agent",
        "test_case_generator": "test_case_generator",
        "execution_agent": "execution_agent",
        "documentation_agent": "documentation_agent",
    },
)

# set the finish points
workflow.set_finish_point("documentation_agent")
workflow.set_finish_point("test_case_generator")
workflow.set_finish_point("debugger_agent")
#workflow.set_finish_point("execution_agent")
# Compile the workflow
app = workflow.compile()

# display the graph
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
