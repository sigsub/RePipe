import pandas as pd
from bs4 import BeautifulSoup
import os
from pathlib import Path
from string import digits
from tqdm import tqdm
import regex as re

def get_content(path_to_html):
    with open(path_to_html, 'r') as f:
        html = f.read()
    return html


def extract_title_from_html(html_content):
    # get the text inside the <title> tag, except data in square brackets
    soup = BeautifulSoup(html_content, "html.parser")
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)
        # Remove text in square brackets
        if '[' in title:
            title = title.split(']')[1].strip()
        return title
    else:
        print("Title tag not found.")
        return None


def extract_abstract_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    abstract_div = soup.find('div', class_='ltx_abstract')
    abstract_text = ""
    if abstract_div:
        paragraph_tags = abstract_div.find_all('p')
        if paragraph_tags:
            text_parts = [p.get_text(strip=True) for p in paragraph_tags]
            abstract_text = " ".join(text_parts)
    return abstract_text


punctuation_pattern = re.compile(r'^[\p{P}\p{S}]+$', re.UNICODE)

def is_not_punctuation_sequence(text):
    # Strip whitespace first
    stripped_text = text.strip()
    if not stripped_text:
        return False
    # Match against the pattern
    return not punctuation_pattern.match(stripped_text)


def extract_all_paragraphs_from_html_v2(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # first, look for divs of the class ltx_para or ltx_para ltx_noindent
    paragraphs = []
    sections = []
    for div in soup.find_all("div"):
        if div.get("class") and "ltx_para" not in div.get("class"):
            continue
        previous_section = div.find_previous("h2")
        if not previous_section:
            continue
        if div.get("id") and "S" in div.get("id"):
            div_texts = []
            for p in div.find_all("p"):
                    # look for <span> tags with ltx_text class
                    for span in p.find_all("span", class_="ltx_text"):
                        if span.get("class") and "ltx_ref_tag" in span.get("class"):
                            continue
                        # get the text inside the <span> tag
                        text = span.get_text(strip=True)
                        if text:
                            div_texts.append(text)
                            # find the section name in which the paragraph is located.
            text = " ".join(div_texts)
            paragraphs.append(text)
            section_name = previous_section.get_text(strip=True).lstrip(digits)
            sections.append(section_name)
    return paragraphs, sections




def extract_all_paragraphs_from_html(html_content, tags_to_remove=['math', 'cite', 'table', 'figure']):
    for t in tags_to_remove:
        html_content = re.sub(rf'<{t}.*?>.*?</{t}>', ' ', html_content, flags=re.DOTALL)
    # sub double spaces with a single space
    html_content = re.sub(r'\s+', ' ', html_content)
    soup = BeautifulSoup(html_content, "html.parser")

            
    # Extract all paragraphs from the body, every paragraph is in a separate <p> tag with a class="ltx_p". each each id=... of a paragraph must have a capital S in it
    paragraphs = []
    sections = []
    for paragraph in soup.find_all("p", class_="ltx_p"):
        # Check if the paragraph has an id with a capital S
        if paragraph.get("id") and "S" in paragraph.get("id"):
            # Extract text from the paragraph
            text = paragraph.get_text(strip=True)
            # replace whitespaces with a single space
            text = re.sub(r'\s+', ' ', text)
            text = ' '.join([t for t in text.split() if is_not_punctuation_sequence(t.strip())])  # Remove punctuation sequences
            # Append to the list of paragraphs
            # find the section name in which the paragraph is located
            prev_header = paragraph.find_previous("h2")
            if not prev_header:
                continue
            section_name = prev_header.get_text(strip=True)
            sections.append(section_name)
            paragraphs.append(text)

    # try a modification - combine paragraphs in the same section if the total length (in terms of actual words) is less than 100
    combined_paragraphs = []
    sections_after_combination = []
    i=0
    while i < len(paragraphs):
        j = i+1
        length = len(paragraphs[i].split())
        # if j == len(paragraphs):
        #     break
        while length < 100 and j < len(paragraphs) and sections[i] == sections[j]:
            length += len(paragraphs[j].split())
            j += 1
        combined_paragraphs.append(" ".join(paragraphs[i:j]))
        sections_after_combination.append(sections[i])
        i = j
    return combined_paragraphs, sections_after_combination
    
    

def text_list_to_sec(text_list, content):
    """
    finds the section name in which the text is MOST LIKELY located in the html content.
    Since the text has been parsed, we search term by term until only one section remains.
    """
    soup = BeautifulSoup(content, "html.parser")
    # split to sections. Each section is in a <section> tag with a class="ltx_section"
    sections = soup.find_all("section", class_="ltx_section")
    # for each section, get the text inside the <h2> tag
    section_titles_to_texts = {}
    for section in sections:
        section_title = section.find("h2")
        if not section_title:
            print("\n\n\nERROR!\n\n\n")
            continue
        section_titles_to_texts[section_title.get_text(strip=True).lstrip(digits)] = section.get_text(strip=True)[len(section_title.get_text(strip=True)):]
    section_list = []
    for text in text_list:
        candidate_sections = []
        for section_title, section_text in section_titles_to_texts.items():
            is_section = True
            for term in text.split():
                if term not in section_text:
                    is_section = False
                    break
            if is_section:
                candidate_sections.append(section_title)
        if len(candidate_sections) == 1:
            section_list.append(candidate_sections[0])
        elif len(candidate_sections) == 0:
            section_list.append("No section found")
        else:
            # print('More than one candidate found for text:\n', text)
            # print('______________________')
            section_list.append(candidate_sections[-1])
    return section_list
            
 




def extract_tables_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    tables = []
    
    # Extract tables
    for table_index, figure in enumerate(soup.find_all("figure", class_="ltx_table")):
        table_tag = figure.find("table")
        if not table_tag:
            continue
        
        # Extract caption
        caption_tag = figure.find("figcaption")
        caption = caption_tag.get_text(strip=True) if caption_tag else None

        # Extract headers
        headers = []
        header_row = table_tag.find("thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

        # Extract rows
        rows = []
        for row in table_tag.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if cells:  # Avoid empty rows
                rows.append(cells)

        expected_table_ref = f"Table {table_index + 1}"
        references = []
        for paragraph in soup.find_all("p"):
            if any(ref_tag.get("title", "").startswith(expected_table_ref) for ref_tag in paragraph.find_all("a", class_="ltx_ref")):
                references.append(paragraph.get_text(strip=True))

        # Store table with index
        table_data = {
            "index": table_index,
            "caption": caption,
            "headers": headers,
            "rows": rows,
            "references": references
        }

        tables.append(table_data)


    return tables

def main():
  pass

if __name__ == '__main__':
    main()