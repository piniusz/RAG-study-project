from src.agents.pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
import supabase
import os
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai.client import AsyncClient
import asyncio
import logging
import chromadb

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\micha\Documents\ai agents tutorial\RAG\data\03_embedded"
)
deps = PydanticAIDeps(chromadb_client=chroma_client, genai_client=client)


async def call_agent(user_input: str, messages_history: list = None):
    """Call the agent with the user input and return the response."""
    logging.info(f"User input: {user_input}")
    response = await pydantic_ai_expert.run(
        user_input,
        message_history=messages_history,
        deps=deps,
    )
    return response


async def main():
    user_input = input()
    message_history = None

    while user_input != r"/quit":
        if message_history:
            user_input = input()
        llm_response = await call_agent(user_input, message_history)
        print(llm_response.output)
        message_history = llm_response.new_messages()


if __name__ == "__main__":
    asyncio.run(main())
