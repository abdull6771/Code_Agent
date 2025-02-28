
# ui.py
import gradio as gr
import pandas as pd
import os
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from workflows.workflow import app# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

def agent_interface(user_input):
    """Process user input through the LangGraph agent workflow."""
    inputs = {"query": user_input}
    result = app.invoke(inputs)
    return result['code']

# Create Gradio Interface
demo = gr.Interface(
    fn=agent_interface,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your coding query..."),
    ],
    outputs=gr.Textbox(),
    title="AI-Powered Code Agent",
    description="Generate, debug, optimize, and secure code using an AI agent."
)

demo.launch(debug=True, share=True)
