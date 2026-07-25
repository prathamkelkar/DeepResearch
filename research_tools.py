import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import requests
from tavily import TavilyClient
import wikipedia
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup
import html2text
from typing import Optional, List
import io
import random
import time
from contextlib import redirect_stdout
import traceback
import re

# Initializing environment and loading variables
load_dotenv()

# Setting up user-agent for requests to arXiv
session = requests.Session()
session.headers.update({
    "User-Agent": "LF-ADP-Agent/1.0 (mailto: kelkarpratham27@gmail.com)"
})

def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    A function to search arXiv for research papers that match the given query.

    Arguments:
        query (str): The search query.
        max_results (int): Number of results to return (default - 5).
    
    Returns:
        results (list[dict]): A list of dictionaries with keys : 'title', 'authors', 'published', etc.
    """

    url = f"https://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results={max_results}"

    # fetching raw xml results
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    
    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

    # extracting the relevant source information and formatting it into a list of dictionaries
    try:
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text[:10]
            url_abstract = entry.find('atom:id', ns).text
            summary = entry.find('atom:summary', ns).text.strip()

            # attempting to find a link to the paper listed
            link_pdf = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    link_pdf = link.attrib.get('href')
                    break

            # storing each part of the information about the source as a key-value pair
            results.append({
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abstract,
                "summary": summary,
                "link_pdf": link_pdf
            })
        
        return results
    
    except Exception as e:
        return[{"error": f"parsing failed: {str(e)}"}]

# JSON function definition to pass into the LLM call
arxiv_search_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "A function to search arXiv for research papers that match the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

def tavily_search_tool(query: str, max_results: int = 10) -> list[dict]:
    """
    A function to perform a web search using the Tavily web search engine API.

    Arguments:
        query (str): The search query.
        max_results (int): Number of results to return (default - 10).
    
    Returns:
        results (list[dict]): A list of dictionaries with keys : 'title', 'content', 'url', etc.
    """

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")
    
    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results
        )
        
        results = []
        # formatting results into a list of dictionaries
        for x in response.get("results", []):
            results.append({
                "title": x.get("title", ""),
                "content": x.get("content", ""),
                "url": x.get("url", "")
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e)}]

# JSON function definition to pass into the LLM call
tavily_search_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

def wikipedia_search_tool(query: str, sentences: int = 10) -> list[dict]:
    """
    A function to perform a web search using the Tavily web search engine API.

    Arguments:
        query (str): The search query.
        sentences (int): Number of sentences that will be included in the summary (default - 10)
    
    Returns:
        results (list[dict]): A list of dictionaries with keys : 'title', 'summary', 'url', etc.
    """

    results = []
    try:
        page_title = wikipedia.search(query)[0]
        page = wikipedia.page(page_title)
        summary = wikipedia.summary(page_title, sentences=sentences)

        # formatting the search results into a list of dictionaries
        results = [{
            "title": page.title,
            "summary": summary,
            "url": page.url
        }]

        return results
    
    except Exception as e:
        return [{"error": str(e)}]

# json function definition for the LLM call
wikipedia_search_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article."
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


def semantic_scholar_search_tool(query: str, max_results: int = 10) -> list[dict]:
    """
    A function to search peer-reviewed academic literature using the Semantic Scholar API.
    Covers all academic disciplines (not just CS/physics/math like arXiv) and includes
    citation counts as a credibility signal. Use this for broader academic coverage,
    or when a query falls outside arXiv's preprint-focused fields.

    Arguments:
        query (str): The search query.
        max_results (int): Number of results to return (default - 10, max - 100).

    Returns:
        results (list[dict]): A list of dictionaries with keys: 'title', 'abstract',
            'url', 'year', 'citation_count', 'authors', 'venue'.
    """

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # defining a dictionary of search filters to perform a targeted search
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,url,year,citationCount,authors,venue"
    }

    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "DeepResearchAgent/1.0 (mailto:kelkarpratham27@gmail.com)",
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY
    })

    # fetching search results
    try:
        response = session.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []

        # formatting the results into a list of dictionaries
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            results.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "") or "",
                "url": paper.get("url", ""),
                "year": paper.get("year", ""),
                "citation_count": paper.get("citationCount", 0),
                "authors": authors,
                "venue": paper.get("venue", "")
            })

        return results

    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

# json function definition for the LLM to understand when passed into an LLM call
semantic_scholar_search_tool_def = {
    "type": "function",
    "function": {
        "name": "semantic_scholar_search_tool",
        "description": """A function to search peer-reviewed academic literature using the Semantic Scholar API. Covers all academic disciplines (not just CS/physics/math like arXiv) and includes citation counts as a credibility signal. Use this for broader academic coverage, or when a query falls outside arXiv's preprint-focused fields.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the semantic scholar paper."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to fetch",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

def scrape_webpage_single(url: str, keywords: Optional[List[str]] = None) -> dict:
    """
    A function to safely web-scrape content from a URL.

    Arguments:
        url (str): The url used for scraping
        keywords (list): optional list of keywords to focus the content extraction

    Returns:
        result: formatted webpage content as text
    """

    # check if url is valid
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"Error: Invalid URL format: {url}. Provide a valid URL"
    
        # Block potentially dangerous URLs
        blocked_domains = [
            "localhost", "127.0.0.1", "0.0.0.0", 
            "192.168.", "10.0.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
            "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", 
            "172.28.", "172.29.", "172.30.", "172.31."
        ]

        if any(domain in parsed_url.netloc for domain in blocked_domains):
            return f"Error: Acess to internal/local URLs is blocked for security: {url}"

        print(f"Scraping URL: {url}")

        # setting headers that mimic a real browser
        headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

        # setting a reasonable timeout
        timeout = 10

        # make the request
        response = requests.get(url, headers=headers, timeout=timeout)

        # check if request was successful
        if response.status_code != 200:
            if response.status_code == 403:
                return f"Error: Access Forbidden (403). The website is actively blocking web-scrapers."
            return f"Error: Failed to fetch the webpage. Status code: {response.status_code}"

        #using beautifulsoup to parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'iframe', 'footer', 'nav', 'header', 'aside', 'form', 'noscript', 'meta', 'link']):
            element.decompose()

        title = soup.title.string if soup.title else "No title found"

        # extracting main content areas

        # trying to find the main content first
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')

        # if the main content is not found then use the body content
        if not main_content:
            main_content = soup.body

        # converting to plain text with specific settings
        # Convert to plain text with specific settings
        h = html2text.HTML2Text()
        h.ignore_links = True  # ignoring links to reduce noise
        h.ignore_images = True
        h.ignore_tables = False
        h.unicode_snob = True
        h.body_width = 0

        if main_content:
            text_content = h.handle(str(main_content))
        else:
            text_content = h.handle(response.text)

        # cleaning up the text content and removing extra whitespace
        text_content = ' '.join(text_content.split())

        # extracting relevant content based on keywords if provided
        if keywords:
            # splitting content into paragraphs
            paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]

            # scoring each paragraph based on keyword presence
            scored_paragraphs = []

            for paragraph in paragraphs:
                score = 0
                for keyword in keywords:
                    if keyword.lower() in paragraph.lower():
                        score += 1
                if score > 0:
                    scored_paragraphs.append((paragraph, score))
            
            # taking top paragraphs by score
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

            # Taking paragraphs with highest scores but limiting content
            selected_paragraphs = []
            total_length = 0
            max_content_length = 8000

            for paragraph, score in scored_paragraphs:
                if total_length + len(paragraph) <= max_content_length:
                    selected_paragraphs.append(paragraph)
                    total_length += len(paragraph)
                else:
                    # If the whole para can't be fit, try to find a good breaking point
                    remaining_length = max_content_length - total_length
                    if remaining_length > 100:  # Only break if we have enough space for meaningful content
                        break_point = paragraph[:remaining_length].rfind('.')
                        if break_point > remaining_length * 0.8:  # If we can find a good sentence break
                            selected_paragraphs.append(paragraph[:break_point + 1])
                            total_length += break_point + 1
                    break
            
            # joining the selected paras
            text_content = '\n\n'.join(selected_paragraphs)

            if total_length >=max_content_length:
                text_content += "\n\n[Content truncated due to big length...]"

        # If no keywords were provided or no matches were found, use the og content with the length limit
        else:
            max_content_length = 8000
            if len(text_content) > max_content_length:
                # trying to find a good breakpoint
                break_point = text_content[:max_content_length].rfind('.')
                if break_point > max_content_length * 0.8:
                    text_content = text_content
                else:
                    text_content = text_content[:max_content_length]
                text_content += "\n\n[Content truncated due to length. Try using a different search method like Tavily search instead or use other key words or phrases.]"


        # format the final response
        result = {"title": title, "url": url, "content": text_content}

        return result

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while trying to access {url}"
    
    except requests.exceptions.ConnectionError:
        return f"Error: Failed to connect to {url}. The site might be down or the URL might be incorrect."
    
    except requests.exceptions.RequestException as e:
        return f"Error requesting {url}: {str(e)}"
    
    except Exception as e:
        return f"Error scraping webpage {url}: {str(e)}"


def scrape_webpage_from_url_list_tool(urls: list, keywords: Optional[List[str]] = None) -> list[dict]:
    """
    A function to safely web-scrape content from a list of URLs.

    Arguments:
        urls (list): The URLs to scrape.
        keywords (list): Optional list of keywords applied to every URL
            to focus content extraction.

    Returns:
        results (list[dict]): One result dict per URL.
    """

    results = []

    for url in urls:
        if keywords:
            results.append(scrape_webpage_single(url, keywords=keywords))
        else:
            results.append(scrape_webpage_single(url))

    return results

# JSON function definition to pass into the LLM call
scrape_webpage_from_url_list_tool_def = {
    "type": "function",
    "function": {
        "name": "scrape_webpage_from_url_list_tool",
        "description": """A function to safely web-scrape content from a list of URLs.""",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The URLs to scrape."
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of keywords applied to every URL to focus content extraction."
                }
            },
            "required": ["urls"]
        }
    }
}



def test_python_execution(code: str) -> dict:
    """
    A function to test python code and diagnose issues.

    Arguments:
        code (str): The code that is to be tested

    Returns:
        output_dict (dict): a dictionary containing the output text of the executed code and the local variables created during execution.
        
    """

    # creating a global environment
    test_globals = {
        'random': random,
        'randint': random.randint,
        'time': time,
        'sleep': time.sleep,
        '__name__': '__main__',
    }

    # creating an empty locals dictionary
    test_locals = {}

    # Capturing output
    output = io.StringIO()

    # Executing with detailed error reporting
    with redirect_stdout(output):
        print(f"Executing code: \n{code}")

        try:
            # trying to compile first to catch syntax errors
            compiled_code = compile(code, '<string>', 'exec')
            print("Compilation successful")

            try:
            
                # trying to execute if compilation was successful
                exec(compiled_code, test_globals, test_locals)

                # checking which local variables were defined
                print(f"Defined locals: {list(test_locals.keys())}")

                # calling the main block directly if the code defines it
                if "__name__" in test_globals and test_globals["__name__"] == "__main__":
                    print("Running main block...")

            except Exception as e:
                print(f"Runtime error: {type(e).__name__}: {str(e)}")
                # getting the traceback information
                traceback.print_exc(file=output)

        except Exception as e:
            print(f"Syntax error: {str(e)}")

    output_text = output.getvalue()

    # creating a dictionary for storing all the variables defined by the LLM in its code
    safe_locals = {}
    for key, value in test_locals.items():
        if key.startswith("__"):
            continue
        if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
            safe_locals[key] = value
        else:
            safe_locals[key] = f"<{type(value).__name__} object>"

    output_dict = {"output_text": output_text, "local_vars": safe_locals}

    return output_dict

def run_python_code_tool(raw_input_text: str) -> dict:
    """
        A function to safely run Python code using an external Python process.
    
        Arguments:
            raw_input_text (str): The raw input text that contains the code that will be executed, formatted with <python> and </python> tags to mark the star and the end respectively
    
        Returns:
            output_dict (dict): a dictionary containing the output text of the executed code and the local variables created during execution.
            
        """

    pattern = r"(?<=<python>).*?(?=</python>)"

    code = ""

    matchobject = re.search(pattern, raw_input_text, re.DOTALL)
    if matchobject:
        code = matchobject.group(0)

    if not code:
        return {"error": "Input not properly formatted with <python> and </python> tags", "output_text": [], "local_vars": {}}
    

    # checking for potentially dangerous operations
    dangerous_operations = [
        "os.system", "os.popen", "os.unlink", "os.remove",
        "subprocess.run", "subprocess.call", "subprocess.Popen",
        "shutil.rmtree", "shutil.move", "shutil.copy",
        "open(", "file(", "eval(", "exec(", 
        "__import__", "input(", "raw_input(",
        "__builtins__", "globals(", "locals(",
        "compile(", "execfile(", "reload("
    ]

# Safe imports that should be allowed
    safe_imports = {
        "import datetime", "import math", "import random", 
        "import statistics", "import collections", "import itertools",
        "import re", "import json", "import csv", "import numpy",
        "import pandas", "from math import", "from datetime import",
        "from statistics import", "from collections import",
        "from itertools import", "from random import", "from random import randint",
        "from random import choice", "from random import sample", "from random import random",
        "from random import uniform", "from random import shuffle", "import time",
        "from time import sleep", "import numpy", "import pandas"
    }

    # checking for dangerous operations
    for dang_op in dangerous_operations:
        if dang_op in code:
            return {"error": f"Code contains potentially unsafe operation: {dang_op}", "output_text": [], "local_vars": {}}

    # checking each line for imports
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            is_safe = any(line.startswith(safe_import) for safe_import in safe_imports)
            if not is_safe:
                return {"error": f"Code contains potentially unsafe import: {line}", "output_text": [], "local_vars": {}}

    # Executing using the helper function test_python_execution
    test_result = test_python_execution(code)

    # extracting the relevant output from the execution
    cleaned_text_output = []
    for line in test_result["output_text"].splitlines():
        if line.startswith("Executing code:") or line.startswith("Compilation successful") or  line.startswith("Execution successful") or "Defined locals:" in line:
            continue
        cleaned_text_output.append(line)

    test_result["output_text"] = cleaned_text_output
    test_result["error"] = None
    return test_result

# JSON function definition to pass into the LLM call
run_python_code_tool_def = {
    "type": "function",
    "function": {
        "name": "run_python_code_tool",
        "description": """A function to safely run Python code using an external Python process.""",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_input_text": {
                    "type": "string",
                    "description": "The raw input text that contains the code that will be executed, formatted with <python> and </python> tags to mark the star and the end respectively"
                }
            },
            "required": ["raw_input_text"]
        }
    }
}


# defining a dictionary of all tools to be used
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
    "scrape_webpage_from_url_list_tool": scrape_webpage_from_url_list_tool,
    "run_python_code_tool": run_python_code_tool,
    "semantic_scholar_search_tool": semantic_scholar_search_tool
}