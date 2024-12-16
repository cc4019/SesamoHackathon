import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def fetch_arxiv_papers(query, max_results=50, start_index=0):
    """Fetch papers from arXiv based on a query."""
    base_url = "http://export.arxiv.org/api/query"
    # query = f"{query} AND (\"causal inference\" OR \"causal discovery\" OR \"heterogeneous treatment effect\")"
    
    query = f"{query}"
    params = {
        "search_query": query,
        "start": start_index,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching papers: {response.status_code}")
        return None

def parse_arxiv_response(xml_response):
    """Parse the arXiv API XML response to extract relevant metadata."""
    root = ET.fromstring(xml_response)
    namespace = "{http://www.w3.org/2005/Atom}"
    papers = []

    for entry in root.findall(f".//{namespace}entry"):
        paper = {
            "title": entry.find(f"{namespace}title").text.strip(),
            "summary": entry.find(f"{namespace}summary").text.strip(),
            "id": entry.find(f"{namespace}id").text.strip(),
            "published": entry.find(f"{namespace}published").text.strip(),
        }
        papers.append(paper)

    return papers

def filter_papers_by_multiple_topics(papers, keyword_groups):
    """Filter papers based on multiple sets of keywords."""
    filtered_papers = {topic: [] for topic in keyword_groups}

    for paper in papers:
        content = (paper["summary"] + paper["title"]).lower()

        for topic, keywords in keyword_groups.items():
            if any(keyword.lower() in content for keyword in keywords):
                filtered_papers[topic].append(paper)
                break  # Avoid duplicating the same paper across topics

    return filtered_papers

def filter_papers_by_recent_time(papers, days=7):
    """Filter papers published in the past specified number of days."""
    recent_papers = []
    current_time = datetime.utcnow()
    time_threshold = current_time - timedelta(days=days)

    for paper in papers:
        published_date = datetime.strptime(paper["published"], "%Y-%m-%dT%H:%M:%SZ")
        if published_date >= time_threshold:
            recent_papers.append(paper)
    return recent_papers

def rank_papers_by_downloads(papers):
    """Rank papers by download count (mock implementation)."""
    for paper in papers:
        paper["downloads"] = hash(paper["id"]) % 100  # Randomized download count

    ranked_papers = sorted(papers, key=lambda x: x["downloads"], reverse=True)
    return ranked_papers

def generate_summary(paper):
    """
    Generate a short summary for a given paper.
    This function uses the paper's title and summary to construct a concise description.
    """
    title = paper.get("title", "No title available").strip()
    abstract = paper.get("summary", "No abstract available").strip()

    # Limit the abstract to the first 3 sentences for brevity
    abstract_sentences = abstract.split(". ")
    short_abstract = ". ".join(abstract_sentences[:3]) + ("..." if len(abstract_sentences) > 3 else "")

    return f"Title: {title}\nSummary: {short_abstract}"

if __name__ == "__main__":
    # Search Keywords
    queries = [
        "abs:(causal OR heterogeneous) AND cat:econ.GN"
    ]

    keywords = [
            "Multimodal AI",
            "Healthcare Applications",
            "Shapley Values",
            "Predictive Analytics",
            "Holistic AI in Medicine"      
    ]
    keywords = [
        'Conversational AI', 
        'Natural Language Processing', 
        'Mental Health', 
        'Interdisciplinary Collaboration.', 
        'Medical Text Summarization', 
        'Large Language Models', 
        'Digital Health', 
        'Patient Engagement',
          'Ethical Considerations', 
          'AI in Healthcare']
    unique_papers = {}
    
    query = " OR ".join([f'all:"{keyword}"' for keyword in keywords])

    # query = 'all:"** multimodal AI, healthcare applications, predictive modeling, electronic health records, Shapley values, data fusion, machine learning, clinical decision-making, patient outcomes, AI interpretability."'
    # query = 'all:"Healthcare Technology." OR all:"Patient Engagement" OR all:"Ethical Considerations" OR all:"Conversational AI" OR all:"** \nLarge Language Models" OR all:"Natural Language Processing" OR all:"Mental Health" OR all:"Chatbots" OR all:"Unstructured Data" OR all:"Digital Health"'
    # print(query)
    for start_index in range(0, 100, 50):  # Paginate up to 100 results
        xml_response = fetch_arxiv_papers(query, max_results=50, start_index=start_index)
        if xml_response:
            papers = parse_arxiv_response(xml_response)
            for paper in papers:
                unique_papers[paper["id"]] = paper

    print(f"Total unique papers fetched: {len(unique_papers)}")

    all_papers = list(unique_papers.values())
    all_papers = {'healthcare': all_papers}

    # topic_filtered_papers = filter_papers_by_multiple_topics(all_papers, keywords)

    for topic, papers in all_papers.items():
        print(f"\nTop Papers for {topic.replace('_', ' ').title()}:")
        
        days_filter = 90 if topic == "marketing_measurement" else 30
        papers = filter_papers_by_recent_time(papers, days=days_filter)

        if not papers:
            print(f"No papers found for {topic.replace('_', ' ').title()}.\n")
            continue

        ranked_papers = rank_papers_by_downloads(papers)
        top_papers = ranked_papers[:50]

        for idx, paper in enumerate(top_papers):
            summary = generate_summary(paper)  # Generate a summary for each paper
            print(f"\n{idx + 1}. {summary}")
            print(f"Published: {paper['published']}")
            print(f"Downloads: {paper['downloads']}")
            print(f"Link: {paper['id']}")