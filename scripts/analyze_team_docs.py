import re
import time
import json
import requests
import logging
import os
import sys
from pathlib import Path
import PyPDF2
import spacy
import xml.etree.ElementTree as ET
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables and set up API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("No OpenAI API key found in environment variables!")
    logging.info("Please ensure you have set OPENAI_API_KEY in your .env file")
    sys.exit(1)

def get_project_paths():
    """Get and display all relevant paths for debugging."""
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    project_root = script_dir.parent
    docs_path = project_root / "data" / "team_docs"
    output_path = project_root / "data" / "analysis_results"
    
    logging.debug(f"Script location: {script_path}")
    logging.debug(f"Project root: {project_root}")
    logging.debug(f"Team docs path: {docs_path}")
    logging.debug(f"Team docs exists: {docs_path.exists()}")
    logging.debug(f"Output path: {output_path}")
    logging.debug(f"Output exists: {output_path.exists()}")

    return docs_path, output_path

def get_external_papers_path():
    """Get the path to the external papers folder."""
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    project_root = script_dir.parent
    external_papers_path = project_root / "data" / "external-paper"
    
    logging.debug(f"External papers path: {external_papers_path}")
    logging.debug(f"External papers path exists: {external_papers_path.exists()}")
    
    return external_papers_path

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF files."""
    logging.debug(f"Starting PDF extraction for: {pdf_path}")
    logging.debug(f"File exists: {pdf_path.exists()}")
    logging.debug(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            logging.debug("Creating PDF reader...")
            pdf_reader = PyPDF2.PdfReader(file, strict=False)
            
            logging.debug(f"PDF loaded. Pages: {len(pdf_reader.pages)}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                logging.debug(f"Processing page {page_num}")
                try:
                    page_text = page.extract_text()
                    text += page_text
                    logging.debug(f"Page {page_num} extracted: {len(page_text)} chars")
                except Exception as e:
                    logging.error(f"Error on page {page_num}: {str(e)}")
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
    
    logging.debug(f"Total text extracted: {len(text)} chars")
    return text

def retrieve_top_k_relevant_arxiv_papers(query, filtered_papers_lst,external_papers_path,top_k=3):
    """Retrieve papers from the relevant Arxiv Papers."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    paper_abstract_lst = [paper['summary'] for paper in filtered_papers_lst]
    paper_embeddings = [model.encode(abstract) for abstract in paper_abstract_lst]

    # Compute query embedding
    query_embedding = model.encode(query)

    # Compute cosine similarity scores
    cosine_similarities = [
        cosine_similarity([query_embedding], [embedding])[0][0]
        for embedding in paper_embeddings
    ]

    print("Top-{top_k} cosine similarity scores:", sorted(cosine_similarities,reverse=True)[:top_k])

    # Get indices of the top-k most similar papers
    top_k_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:top_k]

    retrieved_papers = []
    MAX_RETRIES = 5

    for idx in top_k_indices:
        paper = filtered_papers_lst[idx]
        
        # Convert abstract page URL to PDF download URL
        pdf_url = paper['id'].replace("abs", "pdf")
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', paper['title']).replace(" ", "_")
        pdf_path = Path(external_papers_path) / f"{safe_title}.pdf"

        # Download the PDF with retry on rate limit
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    print(f"PDF '{safe_title}' downloaded successfully.")
                    with open(pdf_path, "wb") as file:
                        file.write(response.content)
                    break  # Success, exit retry loop
                elif response.status_code == 429:
                    print("Rate limit hit. Retrying in 1 second...")
                    time.sleep(1)
                else:
                    print(f"Failed to download PDF. HTTP Status: {response.status_code}")
                    break
            except Exception as e:
                print(f"Error downloading PDF: {e}")
                time.sleep(1)
        else:
            print(f"Failed to download PDF '{safe_title}' after multiple attempts.")
            continue  # Skip to the next paper if retries are exhausted
        # Extract full text from the downloaded PDF
        full_text = extract_text_from_pdf(pdf_path)

        # Append the relevant paper details
        retrieved_papers.append({
            "title": paper['title'],
            "path": pdf_path,
            "snippet": paper['summary'], 
            "full_text": full_text  # Include the extracted full text
        })

    return retrieved_papers


# def retrieve_local_papers(query, external_papers_path):
#     """Retrieve papers from the local external papers folder based on the query."""
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     pdf_files = list(external_papers_path.glob("*.pdf"))
#     paper_texts = []
#     paper_paths = []
#     paper_analysis_results = []
    
#     for pdf_file in pdf_files:
#         text = extract_text_from_pdf(pdf_file)
#         analysis_results = analyze_document_with_llm(text)
#         paper_analysis_results.append(analysis_results)
#         paper_texts.append(text)
#         paper_paths.append(pdf_file)
    
#     # Call the semanticSimilarity function for the query
#     query_embedding = model.encode([query])[0]
    
#     # Call the semanticSimilarity function for each paper text
#     paper_embeddings = []
#     for text in paper_analysis_results:
#         embedding = model.encode([text['full_results']])[0]
#         paper_embeddings.append(embedding)
    
#     # Compute similarity scores using cosine similarity
#     cosine_similarities = []
#     for paper_embedding in paper_embeddings:
#         similarity = cosine_similarity([query_embedding], [paper_embedding])[0][0]
#         cosine_similarities.append(similarity)
    
#     # Get the top 3 most similar papers
#     similar_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:3]
#     retrieved_papers = []

#     for index in similar_indices:
#         retrieved_papers.append({
#             "title": paper_paths[index].stem,
#             "path": paper_paths[index],
#             "snippet": paper_analysis_results[index]  # Include a snippet of the text
#         })
    
#     return retrieved_papers

def analyze_document_with_llm(text):
    """Use LLM to analyze document and generate questions."""
    logging.debug("Starting LLM analysis")
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini", 
            temperature=0.7
        )
        
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=2000,
        #     chunk_overlap=200
        # )
        # chunks = text_splitter.split_text(text)
        # logging.debug(f"Split text into {len(chunks)} chunks")
        
        # combined_text = "\n".join(chunks)
        
        prompt = """Analyze this document excerpt and provide:
        1. Key research areas and concepts
        2. Potential improvement questions
        3. Gaps that could be addressed by external research
        
        Document text:
        {text}
        """
        prompt = """
        Please analyze this academic paper excerpt and provide:

        1. Research Framework & Key Concepts
        - What are the main research questions or hypotheses?
        - Which theoretical foundations or methodologies are used?
        - What core concepts and terminology are central to the work?
        - How do these concepts relate to each other?

        2. Critical Analysis & Improvement Areas
        - What assumptions or limitations are present in the methodology?
        - How could the research design be strengthened?
        - What additional variables or factors could be considered?
        - Are there alternative approaches that might yield better results?

        3. Research Gaps & Extension Opportunities
        - Which important questions remain unexplored?
        - What complementary studies could build on these findings?
        - Are there potential applications not discussed?
        - How could the scope be expanded?

        4. Literature Review Strategy
        FORMAT AS:
        Top Five Primary keywords for paper search: keyword1, keyword2, keyword3, keyword4, keyword5
        Note: Return ONLY keyword lists without explanatory sentences or descriptions.

        Document text:
        {text}

        Please support your analysis with specific examples and quotes from the text where relevant.
        """
        
        try:
            safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
            response = llm.invoke(prompt.format(text=safe_text))
            combined_analysis = response.content
        except Exception as e:
            logging.error(f"Error in LLM analysis: {e}")
            logging.error("This might be due to API key issues or rate limiting")
            combined_analysis = ""
        
        # Extract potential improvement questions and gaps
        # improvement_questions = extract_improvement_questions(combined_analysis)
        # gaps = extract_gaps(combined_analysis)
        
        analysis_results = {
            "full_results": response.content,
            "combined_analysis": combined_analysis,
            # "improvement_questions": improvement_questions,
            # "gaps": gaps,
        }
        
        return analysis_results
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        logging.error("This might be due to API key configuration issues")
        return {"error": "Could not complete LLM analysis"}

def extract_improvement_questions(analysis):
    """Extract potential improvement questions from the analysis."""
    # Enhanced regex patterns for multiple variations
    patterns = [
        r"2\.\s*Critical Analysis & Improvement Areas\s*[:\-]?\s*(.*?)(3\.|$)",  # Primary pattern
        r"Critical Analysis\s*[:\-]?\s*(.*?)(Gaps|$)",  # Alternate phrasing
        r"Improvement Areas\s*[:\-]?\s*(.*?)(3\.|$)"  # Possible alternate title
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.DOTALL)
        if match:
            questions_text = match.group(1).strip()
            logging.debug(f"Improvement questions extracted using pattern: {pattern}")
            break
    else:
        logging.error("Could not find 'Potential Improvement Questions' section.")
        return []

    # Split text into individual questions and clean up
    questions = [q.strip('- ').strip() for q in questions_text.split('\n') if q.strip()]
    logging.debug(f"Extracted {len(questions)} improvement questions.")
    return questions

def extract_gaps(analysis):
    """Extract gaps from the analysis."""
    # Enhanced regex patterns for gaps
    patterns = [
        r"3\.\s*Research Gaps & Extension Opportunities\s*[:\-]?\s*(.*?)(4\.|$)",  # Primary pattern
        r"Gaps Identified\s*[:\-]?\s*(.*?)(Further Research|$)",  # Alternate phrasing
        r"Research Gaps\s*[:\-]?\s*(.*?)(Conclusions|$)"  # Possible alternate title
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.DOTALL)
        if match:
            gaps_text = match.group(1).strip()
            logging.debug(f"Gaps extracted using pattern: {pattern}")
            break
    else:
        logging.warning("Could not find 'Gaps' section.")
        return []

    # Split text into individual gaps and clean up
    gaps = [g.strip('- ').strip() for g in gaps_text.split('\n') if g.strip()]
    logging.debug(f"Extracted {len(gaps)} gaps.")
    return gaps

def extract_keywords(analysis):
    """Extract keywords from the analysis."""
    keyword_section = re.search(
        r"(?:\*\*)?Top Five Primary Keywords for Paper Search:(?:\*\*)?\s*(.*)", 
        analysis, 
        re.DOTALL | re.IGNORECASE
    )
    if keyword_section:
        keywords_text = keyword_section.group(1).strip()
        
        # Handle the list format with dashes (-) or commas
        if ',' in keywords_text:
            keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        else:
            # Handle cases where keywords are listed with dashes (-)
            keywords = re.findall(r"-\s*(.+)", keywords_text)
            
        return keywords
    else:
        logging.error("Could not find 'Keywords' section.")
        raise ValueError("Keywords section not found in the analysis.")
    

def fetch_arxiv_papers(query, max_results=50, start_index=0):
    """Fetch papers from arXiv based on a query."""
    base_url = "http://export.arxiv.org/api/query"
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

def fetch_and_process_papers(keywords, max_results=200):
    """Fetch and process papers based on keywords."""
    unique_papers = {}
    query = " OR ".join([f'all:"{keyword}"' for keyword in keywords])
    print(query)
    for start_index in range(0, max_results, 50):  # Paginate up to max_results
        xml_response = fetch_arxiv_papers(query, max_results=100, start_index=start_index)
        if xml_response:
            papers = parse_arxiv_response(xml_response)
            for paper in papers:
                unique_papers[paper["id"]] = paper

    print(f"Total unique papers fetched: {len(unique_papers)}")

    all_papers = list(unique_papers.values())
    return all_papers

def filter_papers_by_recent_time(papers, days=365):
    """Filter papers published in the past specified number of days."""
    recent_papers = []
    current_time = datetime.utcnow()
    time_threshold = current_time - timedelta(days=days)

    for paper in papers:
        published_date = datetime.strptime(paper["published"], "%Y-%m-%dT%H:%M:%SZ")
        if published_date >= time_threshold:
            recent_papers.append(paper)
    return recent_papers

def explain_how_papers_help_with_llm(paper_title, snippet, questions_and_gaps, api_key):
    """
    Use an LLM to generate an explanation for how a paper addresses specific questions or gaps.
    
    Args:
        paper_title (str): Title of the paper.
        snippet (str): Snippet of the paper content.
        questions_and_gaps (list): List of improvement questions and gaps to address.
        api_key (str): API key for accessing the LLM.
    
    Returns:
        str: Explanation generated by the LLM.
    """
    from langchain_openai import ChatOpenAI

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",  # Use a lightweight model; adjust as needed
            temperature=0.7
        )
        
        # Construct the prompt
        prompt = f"""
        Your are a professional researcher providing guidance on relevant papers for a project team.

        A team is working on a project and has identified the following improvement questions and research gaps:
        {questions_and_gaps}

        The following snippet is from the paper titled '{paper_title}':
        {snippet}

        Please explain evaluate whether this paper might help address the identified questions and gaps. If specific questions or gaps are directly addressed, mention them. Otherwise, make sure to not include the identified questions and gaps in the generated explanation.
        """
        
        # Get the response from LLM
        response = llm.invoke(prompt)
        explanation = response.content.strip()
        return explanation
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return "An error occurred while generating the explanation with the LLM."

def summarize_analysis_with_llm(summary_text, api_key, mode):
    """Use LLM to summarize the analysis results."""

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",  # Use a lightweight model; adjust as needed
            temperature=0.7
        )
        if mode == 'aggregation of analysis':
            # Construct the prompt
            prompt = f"""
            You are a professional researcher tasked with summarizing the content of multiple academic papers.

            The following text contains detailed analysis results from multiple papers:
            {summary_text}

            Please provide a single, cohesive summary that with the existing structure across all papers. Ensure the summary is concise, well-structured, and avoids redundant information.
            """
        elif mode == 'aggregation of external papers recommendation':
            # Construct the prompt
            prompt = f"""
            You are a professional researcher tasked with summarizing the external papers recommendation results of multiple academic papers.

            The following text contains detailed external papers recommendation results from various papers:
            {summary_text}

            Please provide a single, cohesive summary that explains how different papers help each team project, including necessary details and specific examples where relevant.
            """

        # Get the response from LLM
        response = llm.invoke(prompt)
        summary = response.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return "An error occurred while generating the summary with the LLM."
    
def main():
    logging.info("=== Starting PDF Analysis Script ===")
    
    # Get the correct docs_path
    docs_path, output_path = get_project_paths()
        
    # Process each PDF in the folder
    pdf_files = list(docs_path.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    aggregated_analysis = ""
    total_keywords = set()
    for pdf_file in pdf_files:
        logging.info(f"\nAnalyzing {pdf_file.name}...")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            logging.warning(f"No text extracted from {pdf_file.name}")
            continue
        
        # Paper Context Agent
        logging.info("Starting document analysis...")
        
        MAX_TRY = 3
        for i in range(MAX_TRY):
            try:
                analysis_results = analyze_document_with_llm(text)
                top_keywords = extract_keywords(analysis_results["full_results"])
                break
            except Exception as e:
                continue
        else:
            logging.error(f"Failed to analyze document {pdf_file.name} after {MAX_TRY} attempts.")

        total_keywords = list(set(total_keywords).union(set(top_keywords)))
        print(total_keywords)
        # Save analysis results as text
        output_file = output_path / f"{pdf_file.stem}_analysis.txt"
        logging.info(f"Saving analysis results to {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write("DOCUMENT ANALYSIS\n")
                f.write("================\n\n")
                f.write("\nANALYSIS AND QUESTIONS:\n")
                f.write(analysis_results["combined_analysis"])
                # f.write("\n\nIMPROVEMENT QUESTIONS:\n")
                # for question in analysis_results["improvement_questions"]:
                #     f.write(f"- {question}\n")
                # f.write("\nGAPS:\n")
                # for gap in analysis_results["gaps"]:
                #     f.write(f"- {gap}\n")
            logging.info("Analysis results saved successfully")
        except Exception as e:
            logging.error(f"Error saving analysis results: {e}")
        
        # Save full analysis results for summarization
        aggregated_analysis += f"\n\n{analysis_results['full_results']}"

    # Generate a summary of the full analysis results
    internal_paper_summary = summarize_analysis_with_llm(aggregated_analysis, api_key, 'aggregation of analysis')

    # Retrieve external papers based on the keywords
    all_external_papers = fetch_and_process_papers(total_keywords)
    if not all_external_papers:
        raise ValueError("No external papers found based on the provided keywords.")
    print(f"Total retrieved papers before the filter: {len(all_external_papers)}")

    # Filter papers based on recent time (by default, 30 days)
    filtered_papers = filter_papers_by_recent_time(all_external_papers)

    print(f"Total retrieved papers after the release date filter: {len(filtered_papers)}")

    # Retrieve papers for improvement questions and gaps
    external_papers_path = get_external_papers_path()
    retrieved_papers = []

    # TODO: Replace the external_papers_path with the filtered_papers
    papers = retrieve_top_k_relevant_arxiv_papers(internal_paper_summary, filtered_papers,external_papers_path,3)
    retrieved_papers.extend(papers)
    
    analysis_results['retrieved_papers'] = retrieved_papers
    output_file = output_path / f"internal_paper_summarization.txt"
    logging.info(f"Saving analysis results to {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write("Summarized ANALYSIS\n")
            f.write(internal_paper_summary)
        logging.info("Analysis results saved successfully")
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")

    # Save retrieved papers
    recommendation_file = output_path / f"summarized_external-paper_recommendation.txt"
    logging.info(f"Saving retrieved papers to {recommendation_file}")
    try:
        with open(recommendation_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write("EXTERNAL PAPER RECOMMENDATIONS\n")
            f.write("=============================\n\n")
            if analysis_results["retrieved_papers"]:
                for paper in analysis_results["retrieved_papers"]:
                    # Use LLM to generate the explanation
                    explanation = explain_how_papers_help_with_llm(
                        paper_title=paper['title'], 
                        snippet=paper['full_text'], # use full text instead of snippet to generate insights on how paper help with llm
                        questions_and_gaps=extract_improvement_questions(internal_paper_summary) + extract_gaps(internal_paper_summary),
                        api_key=api_key
                    )
                    f.write(f"- {paper['title']}:\n")
                    f.write(f"  {explanation}\n\n")
            else:
                f.write("No relevant papers found.\n")
        logging.info("Retrieved papers saved successfully")
    except Exception as e:
        logging.error(f"Error saving retrieved papers: {e}")

    logging.info("=== PDF Analysis Script Completed ===")
    logging.info("=== Summarize How Externals Could Help the Team ===")

    # Summarize all analysis results into one paragraph
    summary_file = output_path / "external_paper_summarization.txt"
    logging.info(f"Summarizing analysis results into {summary_file}")
    
    try:
        summary_paragraphs = []
        for analysis_file in output_path.glob("*_analysis.txt"):
            with open(analysis_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                summary_paragraphs.append(content)
        
        summary_text = "\n\n".join(summary_paragraphs)
        summary = summarize_analysis_with_llm(summary_text, api_key, mode = 'aggregation of external papers recommendation')
        
        with open(summary_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write("SUMMARY OF ANALYSIS RESULTS\n")
            f.write("===========================\n\n")
            f.write(summary)
        
        logging.info("Summary of analysis results saved successfully")
    except Exception as e:
        logging.error(f"Error summarizing analysis results: {e}")

if __name__ == "__main__":
    main()