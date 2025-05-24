# %%
import pandas as pd
import json
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import asyncio
import IPython
import nest_asyncio
import google.genai as genai
from dotenv import load_dotenv, find_dotenv
import os
import chromadb
import logging
import time
import google.genai as genai
import json
from typing import List


logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

# Constants
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")
GEMINI_API_KEY = os.environ.get("PAID_GEMINI_API_KEY")
MAX_RETRY_ATTEMPTS = 3
RETRY_INITIAL_WAIT_SECONDS = 10


# Initialize genai_client (ensure this is done only once)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


async def split_markdown_on_characters(
    page_url: str,
    markdown_content: str,
    add_context: bool = False,
    model_name: str = None,
    genai_client: genai.Client = None,
    length_limit: int = 1000,
    overlap: int = 150,
) -> list:

    def create_chunked_content_list(section_text: str) -> list:
        text_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=length_limit, chunk_overlap=overlap, is_separator_regex=False
        )
        content_chunks = text_chunk_splitter.split_text(section_text)
        return [{"content": chunk_content} for chunk_content in content_chunks]

    chunks = create_chunked_content_list(markdown_content)

    if add_context:
        chunk_dict_list = [
            {
                f"chunk_{idx+1}": data["content"],
            }
            for idx, data in enumerate(chunks)
        ]
        contextualized_chunks = await add_context_to_chunks(
            markdown_content, chunk_dict_list, genai_client, model_name
        )
        results = []
        for i in range(len(chunks)):
            my_dict = {
                "content": contextualized_chunks[f"chunk_{i+1}"],
                "metadata": {"url": page_url, "chunk_no": f"chunk_{i+1}"},
            }
            results.append(my_dict)

    return results


async def add_context_to_chunks(
    full_document: str,
    chunk_dicts: List[dict],
    genai_client: genai.Client,
    model_name: str,
    retry_count: int = 0,
) -> dict:

    DOCUMENT_CONTEXT_PROMPT = f"""
    <document>
    {full_document}
    </document>
    """

    # Create a dictionary to hold chunk content with keys as chunk numbers to easily access them
    chunks_by_id = {}
    for chunk_dict in chunk_dicts:
        chunks_by_id.update(chunk_dict)

    def prepare_chunks_context_prompt(content_dict: dict) -> str:
        chunks_content = ""
        for chunk_id, content in content_dict.items():
            chunks_content += f"\n<{chunk_id}>\n{content}\n</{chunk_id}>\n"
        chunks_context_prompt = f"""
             Here is the chunk we want to situate within the whole document
            <chunk>
            {chunks_content}
            </chunk>

            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
            Iterate through each chunk, generate succinct context and nothing else. Return the context in JSON format:
            {{
                "chunk_contexts": [
                    {{
                        "chunk_no": "chunk_1",
                        "context": "brief context here"
                    }},
                    {{
                        "chunk_no": "chunk_2",
                        "context": "brief context here"
                    }}
                    // and so on for all chunks
                ]
            }}
        """
        return chunks_context_prompt

    user_prompt = prepare_chunks_context_prompt(chunks_by_id)
    output_schema = {
        "type": "object",
        "properties": {
            "chunk_contexts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_no": {"type": "string"},
                        "context": {"type": "string"},
                    },
                },
            }
        },
    }

    async def get_context_list(prompt_text: str) -> str:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config={
                    "system_instruction": DOCUMENT_CONTEXT_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": output_schema,
                    "temperature": 0.5,
                },
            )
            response_text = response.text
            response_json = json.loads(response_text)
            context_list = response_json["chunk_contexts"]
            return context_list
        except Exception as e:
            current_retry = retry_count + 1
            if current_retry < MAX_RETRY_ATTEMPTS:
                wait_time = RETRY_INITIAL_WAIT_SECONDS * (2**current_retry)
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await get_context_list(prompt_text)

    context_list = await get_context_list(user_prompt)

    context_by_chunk_id = {
        f'{item["chunk_no"]}': item["context"] for item in context_list
    }
    missing_chunk_ids = set(chunks_by_id.keys()) - set(context_by_chunk_id.keys())

    if missing_chunk_ids:
        missing_chunks = {
            chunk_id: chunks_by_id[chunk_id] for chunk_id in missing_chunk_ids
        }
        missing_chunks_prompt = prepare_chunks_context_prompt(missing_chunks)
        missing_context_list = await get_context_list(missing_chunks_prompt)
        context_by_chunk_id.update(
            {
                f'{item["chunk_no"]}': item["context"]
                for item in missing_context_list["chunk_contexts"]
            }
        )

    enriched_chunks = {}
    for chunk_id, content in chunks_by_id.items():
        context = context_by_chunk_id.get(chunk_id, "")
        enriched_chunks[chunk_id] = f"<context>{context}</context>\n{content}"

    return enriched_chunks


# %%
if __name__ == "__main__":
    document_path = r"C:\Users\micha\Documents\ai agents tutorial\RAG study project\data\01_raw\pydantic_ai.json"
    with open(document_path, "r", encoding="utf-8") as file:
        url_articles = json.load(file)
        url_articles = [url_articles[68]]
    contents = [i["content"] for i in url_articles]
    api_key = os.environ.get("PAID_GEMINI_API_KEY")
    model = "gemini-2.5-flash-preview-04-17"
    client = genai.Client(api_key=api_key)

    split_tasks = [
        split_markdown_on_characters(
            article["url"], article["content"], True, model, client
        )
        for article in url_articles
    ]
    nest_asyncio.apply()
    split_results = asyncio.run(split_tasks[0])
    flattened_chunks = [chunk for chunk_list in split_results for chunk in chunk_list]
    # chunks = [
    #     {
    #         "chunk_no": k,
    #         "content": i["content"],
    #     }
    #     for k, i in enumerate(flattened_chunks)
    # ]

    # chunks_content = ""
    # for i in chunks:
    #     chunk_no = i["chunk_no"]
    #     chunk = i["content"]
    #     chunks_content += f"\n<chunk_{chunk_no}>\n{chunk}\n</chunk_{chunk_no}>\n"
    # chunks_context_prompt = f"""
    #     Here is the chunk we want to situate within the whole document
    #     <chunk>
    #     {chunks_content}
    #     </chunk>

    #     Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    #     Iterate through each chunk, generate succinct context and nothing else. Return the context in a list format [chunk1_context, chunk2_context, ...].
    #     Ensure that all the contexts are in the same order as the chunks.
    # """
    # # %%
    # context_list = {
    #     "chunk_contexts": [
    #         {
    #             "chunk_no": "chunk_0",
    #             "context": "Instructions on how to run the weather agent example, including setting up API keys and executing the Python script via pip or uv.",
    #         },
    #         {
    #             "chunk_no": "chunk_1",
    #             "context": "Start of the `pydantic_ai_examples/weather_agent.py` code, defining imports, configuring logfire, and the `Deps` dataclass for API client and keys.",
    #         },
    #         {
    #             "chunk_no": "chunk_2",
    #             "context": "Continuation of the `pydantic_ai_examples/weather_agent.py` code, defining the `weather_agent` with its instructions and the `get_lat_lng` tool function, including dummy response fallback.",
    #         },
    #         {
    #             "chunk_no": "chunk_3",
    #             "context": "Continuation of the `get_lat_lng` tool's implementation, including API call to Mapbox and error handling, followed by the definition of the `get_weather` tool with its dummy response fallback.",
    #         },
    #         {
    #             "chunk_no": "chunk_4",
    #             "context": "Implementation of the `get_weather` tool, detailing the API call to tomorrow.io and logging the response.",
    #         },
    #         {
    #             "chunk_no": "chunk_5",
    #             "context": "Continuation of the `get_weather` tool, parsing the API response to extract temperature and description, using a `code_lookup` dictionary for weather codes.",
    #         },
    #         {
    #             "chunk_no": "chunk_6",
    #             "context": "The `main` asynchronous function demonstrating how to initialize the `weather_agent` with API keys and run a query for weather in specific locations.",
    #         },
    #         {
    #             "chunk_no": "chunk_7",
    #             "context": "Introduction to building a Gradio UI for the weather agent, including instructions on how to install Gradio and run the UI script.",
    #         },
    #         {
    #             "chunk_no": "chunk_8",
    #             "context": "Start of the `pydantic_ai_examples/weather_agent_gradio.py` code, detailing imports, Gradio installation check, mapping tool names, and initializing API clients and dependencies.",
    #         },
    #         {
    #             "chunk_no": "chunk_9",
    #             "context": "The `stream_from_agent` asynchronous function, which handles streaming responses from the weather agent for the Gradio UI, including processing tool calls and displaying metadata.",
    #         },
    #         {
    #             "chunk_no": "chunk_10",
    #             "context": "Continuation of `stream_from_agent` showing how tool outputs are appended to the chatbot history and the implementation of `handle_retry` for re-running failed prompts.",
    #         },
    #         {
    #             "chunk_no": "chunk_11",
    #             "context": "The `undo` function for Gradio chatbot, allowing users to revert to a previous state, and the `select_data` helper function for example selection.",
    #         },
    #         {
    #             "chunk_no": "chunk_12",
    #             "context": "The Gradio UI layout definition within `gr.Blocks()`, including HTML headers, chatbot component with examples, and the prompt text box.",
    #         },
    #         {
    #             "chunk_no": "chunk_13",
    #             "context": "Gradio UI interactions, linking prompt submission to the `stream_from_agent` function, and setting up retry and undo functionalities for the chatbot.",
    #         },
    #     ]
    # }
    # # %%
    # chunks_dict = {f'chunk_{int(i["chunk_no"])+1}': i["content"] for i in chunks}
    # context_dict = {
    #     f'{i["chunk_no"]}': i["context"] for i in context_list["chunk_contexts"]
    # }
    # missing_keys = set(chunks_dict.keys()) - set(context_dict.keys())
    # missing_chunks = {key: chunks_dict[key] for key in missing_keys}

    # missing_contexts = {
    #     "chunk_contexts": [
    #         {
    #             "chunk_no": "chunk_0",
    #             "context": "Instructions on how to run the weather agent example, including setting up API keys and executing the Python script via pip or uv.",
    #         },
    #         {
    #             "chunk_no": "chunk_14",
    #             "context": "Continuation of the Gradio UI layout definition, including the chatbot component and examples for user interaction.",
    #         },
    #         {
    #             "chunk_no": "chunk_15",
    #             "context": "Continuation of the Gradio UI layout definition, including the submit button and the `gr.Markdown` component for displaying instructions.",
    #         },
    #     ]
    # }
    # context_dict.update(
    #     {f'{i["chunk_no"]}': i["context"] for i in missing_contexts["chunk_contexts"]}
    # )
    # missing_keys = set(chunks_dict.keys()) - set(context_dict.keys())
    # for chunk, content in chunks_dict.items():
    #     context = context_dict.get(chunk, "")
    #     chunks_dict[chunk] = f"{context} \n\n {content}"
# %%
