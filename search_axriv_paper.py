import requests
import json
import openai
import os 
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

API_KEY = "sk-"

MODEL_TYPE = "gpt-3.5-turbo"

SYS_PROMPT = "As an AI, your task is to evaluate the relevance of academic papers to specified topics. After reading the abstract of a paper and the description of a topic, you are to score the relevance of the paper to the topic on a scale from 0 to 100. A score of 0 means the paper has no relevance to the topic whatsoever, while a score of 100 indicates the paper is highly relevant to the topic. Your score should reflect not just the presence of keywords, but the depth of content alignment with the topic. Please provide just the score as your response, without additional commentary or justification."

ARXIV_API_URL = "http://export.arxiv.org/api/query?sortBy=submittedDate&sortOrder=descending"

def search_arxiv(topic: list[str], max_results: int = 50) -> list:
    query = '+AND+'.join(f'all:{word}' for word in topic)
    search_query = f"{ARXIV_API_URL}&search_query={query}&start=0&max_results={max_results}"
    response = requests.get(search_query)
    root = ET.fromstring(response.content)

    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('arxiv:entry', namespace)
    papers = []
    for entry in tqdm(entries, desc="Searching papers"):
        title = entry.find('arxiv:title', namespace).text.strip()
        summary = entry.find('arxiv:summary', namespace).text.strip()
        id_url = entry.find('arxiv:id', namespace).text.strip()
        pdf_url = id_url.replace('abs', 'pdf') + ".pdf"

        papers.append({
            "title": title,
            "abstract": summary,
            "url": pdf_url
        })
    return papers


def filter_paper(papers: list[dict[str,str]], topic: str):
    client = openai.Client(api_key=API_KEY)
    for paper in tqdm(papers, desc="Filtering papers"):
        response: str = client.chat.completions.create(
            model=MODEL_TYPE,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"Topic: {topic} \n Abstract: {paper['abstract']}"},
            ]       
        ).choices[0].message.content
        
        score = re.search(r"\d+", response)
        try:
            paper["score"] = int(score.group())
        except:
            print("Error: ", response, "title: ", paper["title"])
            continue
    return papers

def download(filename: str, destination: str, bar: int = 50):
    with open(filename, "r") as f:
        papers = json.load(f)
    if not os.path.exists(destination):
        os.makedirs(destination)
    for i,paper in tqdm(enumerate(papers), desc="Downloading papers"):
        if paper["score"] < bar:
            continue
        response = requests.get(paper["url"])
        with open(f"{destination}/{i}.pdf", "wb") as f:
            f.write(response.content)
    print("Download completed.")


def main():
    papers = search_arxiv(["Agent", "serve", "system"], max_results=10)
    result = filter_paper(papers, "Serving system for agent applications.")
    with open("result_score.json", "w") as f:
        json.dump(result, f, indent=4)
    download("result_filter.json", "papers")

if __name__ == "__main__":
    main()
