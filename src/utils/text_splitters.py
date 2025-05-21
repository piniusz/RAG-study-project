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
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.genai as genai
from dotenv import load_dotenv, find_dotenv
import os
import chromadb
import logging
import time
from typing import List
from google.genai.errors import ClientError
import google.genai as genai
from chromadb import ClientAPI as ChromadbClientAPI


logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

# Constants
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")
API_KEY = os.environ.get("PAID_GEMINI_API_KEY")
MAX_RETRIES = 3
INITIAL_WAIT_SECONDS = 10


# Initialize genai_client (ensure this is done only once)
genai_client = genai.Client(api_key=API_KEY)


async def split_markdown_on_h2(page_url: str, markdown_content: str):
    header_definitions = [("##", "Header2")]
    markdown_header_splitter = MarkdownHeaderTextSplitter(header_definitions)
    document_sections = markdown_header_splitter.split_text(markdown_content)

    def create_chunked_content_list(section_text: str) -> list:
        text_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, is_separator_regex=False
        )
        content_chunks = text_chunk_splitter.split_text(section_text)
        return [{"content": chunk_content} for chunk_content in content_chunks]

    processed_sections = []
    for document_section in document_sections:
        section_metadata = {}
        if len(
            document_section.metadata.keys()
        ) == 0 and document_section.page_content.startswith("# "):
            document_title = document_section.page_content[2:]
            continue
        section_heading = document_section.metadata["Header2"]
        section_metadata["url"] = page_url
        section_metadata["tittle"] = document_title
        section_metadata["section"] = section_heading

        text_chunks = create_chunked_content_list(document_section.page_content)
        for chunk_index, chunk in enumerate(text_chunks):
            chunk_metadata = section_metadata.copy()
            chunk_metadata["chunk_no"] = chunk_index
            processed_sections.append(
                {"metadata": chunk_metadata, "content": chunk["content"]}
            )

    return processed_sections


# %%
if __name__ == "__main__":
    file_path = (
        r"C:\Users\micha\Documents\ai agents tutorial\RAG\data\01_raw\pydantic_ai.json"
    )
    with open(file_path, "r") as f:
        contents = json.loads(f.read())
    url = contents[2]["url"]
    content = contents[2]["content"]

    # Check if we're in IPython/Jupyter
    if IPython.get_ipython() is not None:
        nest_asyncio.apply()
    # %%
    section_data = asyncio.run(split_markdown_on_h2(url, content))
    # %%
    tasks = [split_markdown_on_h2(i["url"], i["content"]) for i in contents]

    results = asyncio.run(asyncio.gather(*tasks))
    results_final = [item for sublist in results for item in sublist]
# %%
