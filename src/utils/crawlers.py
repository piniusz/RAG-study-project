# %%
import requests
import markdownify
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Dict, Any, List
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor

# %%
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)


# %%
def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Given a sitemap URL, this function retrieves all URLs listed in the sitemap.
    """
    sitemap_response = requests.get(sitemap_url)
    xml_root = ET.fromstring(sitemap_response.content)
    page_urls = [
        loc.text
        for loc in xml_root.findall(
            ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
        )
    ]
    return page_urls


def extract_markdown_from_urls(page_urls: List[str]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor() as thread_executor:
        markdown_contents = list(
            thread_executor.map(download_url_content_as_markdown, page_urls)
        )
        page_data = [
            {"url": url, "content": content}
            for url, content in zip(page_urls, markdown_contents)
        ]
    return page_data


def download_url_content_as_markdown(url: str):
    session_adapter = HTTPAdapter(max_retries=retries)
    http_session = requests.Session()
    http_session.mount("http://", session_adapter)
    http_session.mount("https://", session_adapter)
    page_response = http_session.get(url)
    if page_response.status_code == 200:
        html_soup = bs(page_response.content, "html.parser")
        article_content = html_soup.find("article")
        return markdownify.markdownify(article_content.prettify(), heading_style="ATX")
    else:
        return None
