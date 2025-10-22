import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def get_content_from_url(url):
    """
    Downloads the HTML content from a given URL and saves it to a file.

    Args:
        url (str): The URL of the webpage to download.
        filename (str): The name of the file to save the HTML content to.
                        Defaults to "downloaded_page.html".
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error downloading HTML: {e}")
        return ""


def content_to_pars(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    articles = soup.find_all('article')
    paragraph_texts = []
    for article in articles:
        # Find all <div> tags with the exact class
        divs = article.find_all('div', class_='primary-cli cli cli-text')
        
        for div in divs:
            # Get all <p> tags inside the div
            paragraphs = div.find_all('p')
            for p in paragraphs:
                paragraph_texts.append(p.get_text(strip=True, separator=' '))
    return paragraph_texts

if __name__ == "__main__":
    # Example usage: with the News Category dataset (Misra, 2022)
    dataset = pd.read_json("LOCATION/OF/YOUR/DATASET", lines=True)
    top_15_cats = dataset['category'].value_counts().head(25).index.tolist()
    dataset = dataset[dataset['category'].isin(top_15_cats)]
    dataset = dataset.groupby('category').sample(n=300, random_state=42).reset_index(drop=True)
    tqdm.pandas(desc="Processing links")
    dataset['pars'] = dataset['link'].progress_apply(lambda x: content_to_pars(get_content_from_url(x)))
    dataset.to_pickle("SAVE/PATH/TO/PKL/FILE")