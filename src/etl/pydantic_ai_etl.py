# %%
import asyncio
import os
import nest_asyncio
from src.utils.crawlers import get_urls_from_sitemap, extract_markdown_from_urls
from src.utils.text_splitters import advanced_markdown_chunking
import logging
import json
from chromadb import ClientAPI as ChromadbClientAPI
import chromadb
import IPython
import src.services.chromaDB as chromaDB
from google import genai
from google.genai import types
from time import time

# %%
chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\micha\Documents\ai agents tutorial\RAG\data\03_embedded"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")


async def main():
    logging.info("Starting Pydantic AI ETL process")
    # Getting raw data from Pydantic AI
    current_directory = os.path.dirname(os.path.abspath(__file__))
    raw_data_directory = os.path.join(current_directory, "..", "..", "data", "01_raw")
    pydantic_dump_path = os.path.join(raw_data_directory, "pydantic_ai.json")

    logging.info("Retrieving URLs from Pydantic AI sitemap.")
    all_pydantic_urls = get_urls_from_sitemap("https://ai.pydantic.dev/sitemap.xml")

    logging.info("Checking for cached content and identifying new URLs")
    known_urls = []
    if os.path.exists(pydantic_dump_path):
        with open(pydantic_dump_path, "r") as file:  #
            cached_content = json.load(file)
            known_urls = [article["url"] for article in cached_content]

    logging.info("Looking for new URLs to process.")
    new_urls = [url for url in all_pydantic_urls if url not in known_urls]
    if new_urls:
        logging.info(f"Processing {len(new_urls)} new URLs")
        url_articles = extract_markdown_from_urls(new_urls)
        url_articles = cached_content + url_articles
        with open(pydantic_dump_path, "w") as file:
            json.dump(url_articles, file, indent=4)
    else:
        url_articles = cached_content

    logging.info("Setting up Gemini client for text splitting")
    api_key = os.environ.get("PAID_GEMINI_API_KEY")
    model = "gemini-2.0-flash"
    client = genai.Client(api_key=api_key)

    logging.info("Splitting article content into chunks")
    split_tasks = [
        advanced_markdown_chunking(
            article["url"], article["content"], True, model, client
        )
        for article in url_articles
    ]

    split_results = await asyncio.gather(*split_tasks)
    flattened_chunks = [chunk for chunk_list in split_results for chunk in chunk_list]

    # %%
    # Loading to ChromaDB
    collection_name = "pydantic_ai_kb"
    logging.info(
        f"Loading {len(flattened_chunks)} documents to ChromaDB collection {collection_name}"
    )
    chroma_client = chromadb.PersistentClient(
        path=r"C:\Users\micha\Documents\ai agents tutorial\RAG\data\03_embedded"
    )
    # drop collection if it exists
    if collection_name in [
        collection.name for collection in chroma_client.list_collections()
    ]:
        logging.info(f"Collection {collection_name} already exists. Deleting it.")
        chroma_client.delete_collection(
            name=collection_name,
        )

    logging.info("Inserting documents into ChromaDB")
    insert_results = chromaDB.load_to_chromadb(
        collection_name=collection_name,
        client=chroma_client,
        data=flattened_chunks,
        batch_size=100,
    )

    logging.info("ETL process completed successfully")
    # get all data with embedding
    return insert_results


# %%
if __name__ == "__main__":
    logging.info("Script execution started")
    if IPython.get_ipython() is not None:
        nest_asyncio.apply()
    result = asyncio.run(main())

# %%
