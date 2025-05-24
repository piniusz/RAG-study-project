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


async def advanced_markdown_chunking(
    page_url: str,
    markdown_content: str,
    context_enriched: bool = False,
    model_name: str = None,
    genai_client: genai.Client = None,
) -> list:
    """Implement semantic + structural chunking"""

    # First: Structural chunking by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_content)

    # Second: Adaptive chunking based on content
    chunks = []
    for split in md_header_splits:
        content = split.page_content
        metadata = split.metadata

        # Adaptive chunk size based on content type
        if any(keyword in content.lower() for keyword in ["example", "code", "usage"]):
            chunk_size = 1500  # Larger for code examples
        elif any(keyword in content.lower() for keyword in ["api", "reference"]):
            chunk_size = 800  # Smaller for API docs
        else:
            chunk_size = 1000  # Default

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        sub_chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(sub_chunks):
            enriched_metadata = {
                **metadata,
                "url": page_url,
                "chunk_index": i,
                "content_type": _classify_content_type(chunk),
                "section_path": " > ".join(metadata.values()) if metadata else "root",
            }
            chunks.append({"content": chunk, "metadata": enriched_metadata})

    if context_enriched:
        enriched_chunks = await add_context_to_chunks(
            full_document=markdown_content,
            chunk_dicts=chunks,
            genai_client=genai_client,
            model_name=model_name,
        )
        return enriched_chunks
    else:
        # If context enrichment is not needed, return the chunks directly
        return chunks


def _classify_content_type(content: str) -> str:
    """Classify chunk content type for better retrieval"""
    content_lower = content.lower()
    if "```" in content or "def " in content or "class " in content:
        return "code_example"
    elif "install" in content_lower or "pip" in content_lower:
        return "installation"
    elif any(
        word in content_lower for word in ["api", "method", "function", "parameter"]
    ):
        return "api_reference"
    elif any(word in content_lower for word in ["example", "tutorial", "guide"]):
        return "tutorial"
    else:
        return "documentation"


async def add_context_to_chunks(
    full_document: str,
    chunk_dicts: List[dict],
    genai_client: genai.Client,
    model_name: str,
    retry_count: int = 3,
) -> dict:
    """Add contextual information to chunks for better search retrieval."""

    # Prepare data structures
    chunks_by_id = _extract_chunks_by_id(chunk_dicts)
    document_context_prompt = _create_document_context_prompt(full_document)
    output_schema = _create_output_schema()

    # Get context for all chunks
    context_by_chunk_id = await _get_contexts_for_chunks(
        chunks_by_id, genai_client, model_name, document_context_prompt, output_schema
    )

    # Create enriched chunks with context
    enriched_chunks = _create_enriched_chunks(chunks_by_id, context_by_chunk_id)

    for i, chunk in enumerate(chunk_dicts):
        chunk_id = f"chunk_{i}"
        if chunk_id in enriched_chunks:
            chunk["content_with_context"] = enriched_chunks[chunk_id]

    for chunk in chunk_dicts:
        content_with_context = chunk["content_with_context"]
        chunk["content"] = content_with_context
        del chunk["content_with_context"]

    return chunk_dicts


def _extract_chunks_by_id(chunk_dicts: List[dict]) -> dict:
    """Extract chunks into a dictionary with chunk IDs as keys."""
    chunks_by_id = {}
    for i, chunk_dict in enumerate(chunk_dicts):
        chunks_by_id["chunk_" + str(i)] = chunk_dict["content"]
    return chunks_by_id


def _create_document_context_prompt(full_document: str) -> str:
    """Create the document context prompt for the AI model."""
    return f"""
    <document>
    {full_document}
    </document>
    """


def _create_output_schema() -> dict:
    """Define the expected JSON schema for the AI response."""
    return {
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


def _prepare_chunks_context_prompt(content_dict: dict) -> str:
    """Prepare the prompt for requesting context for chunks."""
    chunks_content = ""
    for chunk_id, content in content_dict.items():
        chunks_content += f"\n<{chunk_id}>\n{content}\n</{chunk_id}>\n"

    return f"""
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


async def _generate_context_with_retry(
    prompt_text: str,
    genai_client: genai.Client,
    model_name: str,
    document_context_prompt: str,
    output_schema: dict,
    retry_count: int = 3,
) -> List[dict]:
    """Generate context using AI model with retry logic."""
    try:
        response = genai_client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config={
                "system_instruction": document_context_prompt,
                "response_mime_type": "application/json",
                "response_schema": output_schema,
                "temperature": 0.5,
            },
        )
        response_text = response.text
        response_json = json.loads(response_text)
        return response_json["chunk_contexts"]

    except Exception as e:
        current_retry = retry_count + 1
        if current_retry < MAX_RETRY_ATTEMPTS:
            wait_time = RETRY_INITIAL_WAIT_SECONDS * (2**current_retry)
            logging.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return await _generate_context_with_retry(
                prompt_text,
                genai_client,
                model_name,
                document_context_prompt,
                output_schema,
                current_retry,
            )
        raise


async def _get_contexts_for_chunks(
    chunks_by_id: dict,
    genai_client: genai.Client,
    model_name: str,
    document_context_prompt: str,
    output_schema: dict,
) -> dict:
    """Get contexts for all chunks, processing in batches if needed."""

    # Process in smaller batches if there are many chunks
    max_chunks_per_batch = 50  # Reduce batch size
    chunk_items = list(chunks_by_id.items())
    context_by_chunk_id = {}

    for i in range(0, len(chunk_items), max_chunks_per_batch):
        batch_chunks = dict(chunk_items[i : i + max_chunks_per_batch])
        logging.info(
            f"Processing chunk batch {i//max_chunks_per_batch + 1}/{(len(chunk_items)-1)//max_chunks_per_batch + 1}"
        )

        user_prompt = _prepare_chunks_context_prompt(batch_chunks)
        context_list = await _generate_context_with_retry(
            user_prompt,
            genai_client,
            model_name,
            document_context_prompt,
            output_schema,
        )

        batch_context = {
            f'{item["chunk_no"]}': item["context"] for item in context_list
        }
        context_by_chunk_id.update(batch_context)

    return context_by_chunk_id


def _create_enriched_chunks(chunks_by_id: dict, context_by_chunk_id: dict) -> dict:
    """Create enriched chunks by combining original content with context."""
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
        advanced_markdown_chunking(
            article["url"], article["content"], True, model, client
        )
        for article in url_articles
    ]
    nest_asyncio.apply()
    split_results = asyncio.run(split_tasks[0])

# %%
