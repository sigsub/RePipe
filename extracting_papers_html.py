import pandas as pd
import arxiv
import time
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
import requests

client = arxiv.Client()


def get_eprint_link_id(paper_id):
    return f'http://export.arxiv.org/e-print/{paper_id}'

def get_ar5iv_link_id(paper_id):
    return f'https://ar5iv.labs.arxiv.org/html/{paper_id}'

def get_id_from_title(title):
    """
    Downloads the arXiv paper with the closest title match, and prints arXiv ID and authors.

    Args:
        article_name (str): The name of the article to search for.
        download_folder (str, optional): The folder to save the downloaded PDF. Defaults to "arxiv_downloads".
    """
    try:
        # Search arXiv for the article, retrieving up to 10 results
        search = arxiv.Search(query=title, max_results=10)
        results = list(client.results(search))
        # print(results[0].author)
        if not results:
            print(f"No arXiv results found for '{title}'.")
            return
        # Find the closest title match
        best_match = None
        authors = None
        for result in results:
            if title.lower() in result.title.lower(): #Case insensitive comparison.
                best_match = result
                authors = [str(auth) for auth in best_match.authors]
                break  # Stop when a match is found
        if best_match is None:
            print(f"Could not find exact title match for '{title}'.")
            return None
        # Print arXiv ID and authors
        return (best_match.entry_id.split('/')[-1], authors)

    except Exception as e:
        print(f"An error occurred while searching for '{title}': {e}")



    
def download_ar5iv_data(arxiv_id, save_dir, title=None):
    link = get_ar5iv_link_id(arxiv_id)
    response = requests.get(link)
    full_html = response.text
    # check status code
    if response.status_code != 200:
        print(f"Error for : {response.status_code}, status: {response.status_code}")
        return
    if title is not None:
        print(f"Downloading {title} from {link}")
        title = title.replace('/', '$')
        with open(os.path.join(save_dir, f'{arxiv_id}_{title.replace("/", "$")}.html'), 'w') as f:
            f.write(full_html)
    else:
        print(f"Downloading {arxiv_id} from {link}")
        with open(os.path.join(save_dir, f'{arxiv_id}.html'), 'w') as f:
            f.write(full_html)
    return full_html

def download_ar5iv_data_from_title(title, save_dir):
    """
    Downloads the arXiv paper with the closest title match, and prints arXiv ID and authors."""
    out = get_id_from_title(title)
    if out is None:
        return
    else:
        arxiv_id, authors = out
    download_ar5iv_data(arxiv_id, save_dir, title)




def main():
    pass


if __name__ == '__main__':
    main()