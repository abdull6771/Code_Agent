# main.py

import asyncio

from workflows.workflow import app
from langchain_openai import ChatOpenAI
import os

async def run_app():
    # Load environment variables
    os.environ["OPENAI_API_KEY"] = "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"

# Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Run the application
    inputs = {"query": """ Explain the following code inputs = {"query": "write code to find the sum of the numbers in an array"}

    # Invoke the graph with the input
    result = app.invoke(inputs)
    print(result)"""}
    config = {"recursion_limit": 50}
    
    logs = []
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                logs.append(v)
    
    for log in logs:
        print(log)
    #result = app.invoke(inputs)

def main():
    asyncio.run(run_app())

if __name__ == "__main__":
    main()
