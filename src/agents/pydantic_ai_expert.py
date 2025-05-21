# %%
from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
import logfire
import asyncio
import httpx
import os
from google import genai

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
from chromadb import ClientAPI
from typing import List
import nest_asyncio
import chromadb

load_dotenv(find_dotenv())

# check if interacivte mode


@dataclass
class PydanticAIDeps:
    chromadb_client: ClientAPI
    genai_client: genai.Client


llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
model = GeminiModel(GEMINI_MODEL, provider="google-gla")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logfire.configure(send_to_logfire="if-token-present")

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG using retrieve_relevant_chunks tool. But don't put literal user query. Try to rephrase it to be more specific and relevant to the documentation. For example, if the user asks "How to use Pydantic AI?", you can use "Pydantic AI usage examples" as a query.
If provided chunks are not enough, use retrieve_section tool to get the specific section of the documentation.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    model_settings=GeminiModelSettings(temperature=0),
    instrument=True,
)


async def get_embedding(text: str, client: genai.Client) -> List[float]:
    """Get embedding vector from gemini"""

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    return response.embeddings[0].values


@pydantic_ai_expert.tool
async def retrieve_relevant_chunks(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query using RAG.

    Args:
        ctx: The context
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 5 most relevant documentation chunks from the Pydantic AI knowledge base
    """
    try:
        query_embedding_vector = await get_embedding(user_query, ctx.deps.genai_client)
        pydantic_documentation_collection = ctx.deps.chromadb_client.get_collection(
            "pydantic_ai_kb"
        )
        search_results = pydantic_documentation_collection.query(
            query_embeddings=[query_embedding_vector],
            n_results=5,
        )

        # Format the results
        formatted_documentation = query_results_to_markdown(search_results)
        return formatted_documentation

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_expert.tool
async def retrieve_section(
    ctx: RunContext[PydanticAIDeps], metadata_to_match: List[dict]
) -> str:
    """
    Retrieves specific sections of Pydantic documentation based on metadata filters.

    Args:
        ctx: The context with dependencies including ChromaDB client
        metadata_to_match: List of dicts with 'section' and 'url' keys to filter documentation

    Returns:
        Formatted markdown string of the matched documentation sections
    """
    collection = ctx.deps.chromadb_client.get_collection("pydantic_ai_kb")
    or_conditions = []
    for item in metadata_to_match:
        condition = {
            "$and": [
                {"section": {"$eq": item["section"]}},
                {"url": {"$eq": item["url"]}},
            ]
        }
        or_conditions.append(condition)

    where_filter = (
        {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]
    )
    results = collection.get(
        where=where_filter,
        include=[
            "metadatas",
            "documents",
        ],
    )
    formatted_documentation = query_results_to_markdown(results)
    return formatted_documentation


def query_results_to_markdown(query_results: dict) -> str:
    """
    Transforms the result dictionary into a markdown string.

    Args:
        result: The input dictionary with documents and metadatas.

    Returns:
        A string formatted in markdown.
    """
    markdown_string = ""

    # Helper to store processed combinations of tittle and section to avoid repetition
    processed_sections = set()
    metadata_list = query_results["metadatas"]
    document_list = query_results["documents"]

    # check if this is a list of lists
    if isinstance(metadata_list[0], list):
        metadata_list = [item for sublist in metadata_list for item in sublist]
        document_list = [item for sublist in document_list for item in sublist]

    for j in range(len(metadata_list)):
        meta = metadata_list[j]
        doc_content = document_list[j]  # Get the corresponding document chunk

        tittle = meta.get("tittle", "Untitled").strip()
        section = meta.get("section", "Unsectioned").strip()
        url = meta.get("url").strip()

        # Create a unique key for tittle and section combination
        section_key = (tittle, section)

        if section_key not in processed_sections:
            if markdown_string:  # Add a separator if not the first entry
                markdown_string += "\n---\n\n"
            markdown_string += f"[Documentation URL]({url})\n\n"
            markdown_string += f"# [Tittle]{tittle}\n\n"
            markdown_string += f"## [Section]{section}\n\n"
            processed_sections.add(section_key)

        markdown_string += f"{doc_content.strip()}\n\n"

    return markdown_string


# %%
if __name__ == "__main__":
    # Initialize the clients
    chromadb_client = chromadb.PersistentClient(
        path=r"C:\Users\micha\Documents\ai agents tutorial\RAG\data\03_embedded"
    )
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    # Create the dependencies
    @dataclass
    class testingDeps:
        deps: PydanticAIDeps

    deps = testingDeps(
        PydanticAIDeps(chromadb_client=chromadb_client, genai_client=genai_client)
    )
    # %%
    nest_asyncio.apply()
    chunks = asyncio.run(retrieve_relevant_chunks(deps, "Weather Agent example"))
    results = asyncio.run(
        retrieve_section(
            deps,
            [
                {
                    "section": "Example Code",
                    "url": "https://ai.pydantic.dev/examples/weather-agent/",
                }
            ],
        )
    )

# %%
