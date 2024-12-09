import re
import json
import logging
import os
import sys
from pathlib import Path
import PyPDF2
import spacy
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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
    
    logging.debug(f"Script location: {script_path}")
    logging.debug(f"Project root: {project_root}")
    logging.debug(f"Team docs path: {docs_path}")
    logging.debug(f"Team docs exists: {docs_path.exists()}")
    
    return docs_path

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

def extract_keywords(text):
    """Extract domain specific key terms using spaCy."""
    logging.debug("Starting keyword extraction")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        logging.error(f"spaCy model loading error: {e}")
        logging.info("Attempting to install spaCy model...")
        try:
            import subprocess
            subprocess.check_call(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.error(f"Model installation failed: {e}")
            return {}

    doc = nlp(text)
    keywords = []
    for chunk in doc.noun_chunks:
        keywords.append(chunk.text.lower())
    
    keyword_freq = Counter(keywords)
    result = dict(sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    logging.debug(f"Found {len(result)} keywords")
    return result

def retrieve_local_papers(query, external_papers_path):
    """Retrieve papers from the local external papers folder based on the query."""
    pdf_files = list(external_papers_path.glob("*.pdf"))
    paper_texts = []
    paper_paths = []
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        paper_texts.append(text)
        paper_paths.append(pdf_file)
    
    # Compute similarity scores
    vectorizer = TfidfVectorizer().fit_transform([query] + paper_texts)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Get the top 1 most similar papers
    similar_indices = cosine_similarities.argsort()[-1:][::-1]
    retrieved_papers = []
    for index in similar_indices:
        retrieved_papers.append({
            "title": paper_paths[index].stem,
            "path": paper_paths[index],
            "snippet": paper_texts[index][:200]  # Include a snippet of the text
        })
    
    return retrieved_papers

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
        
        try:
            response = llm.invoke(prompt.format(text=text))
            combined_analysis = response.content
        except Exception as e:
            logging.error(f"Error in LLM analysis: {e}")
            logging.error("This might be due to API key issues or rate limiting")
            combined_analysis = ""
        
        # Extract potential improvement questions and gaps
        improvement_questions = extract_improvement_questions(combined_analysis)
        gaps = extract_gaps(combined_analysis)
        
        # Retrieve papers for improvement questions and gaps
        external_papers_path = get_external_papers_path()
        retrieved_papers = []
        papers = retrieve_local_papers(response.content, external_papers_path)
        retrieved_papers.extend(papers)
        
        analysis_results = {
            "combined_analysis": combined_analysis,
            "improvement_questions": improvement_questions,
            "gaps": gaps,
            "retrieved_papers": retrieved_papers
        }
        
        print(analysis_results)

        return analysis_results
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        logging.error("This might be due to API key configuration issues")
        return {"error": "Could not complete LLM analysis"}

def extract_improvement_questions(analysis):
    """Extract potential improvement questions from the analysis."""
    # Enhanced regex patterns for multiple variations
    patterns = [
        r"2\.\s*Potential Improvement Questions\s*[:\-]?\s*(.*?)(3\.|$)",  # Primary pattern
        r"Improvement Questions\s*[:\-]?\s*(.*?)(Gaps|$)",  # Alternate phrasing
        r"Questions for Improvement\s*[:\-]?\s*(.*?)(3\.|$)"  # Possible alternate title
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.DOTALL)
        if match:
            questions_text = match.group(1).strip()
            logging.debug(f"Improvement questions extracted using pattern: {pattern}")
            break
    else:
        logging.warning("Could not find 'Potential Improvement Questions' section.")
        return []

    # Split text into individual questions and clean up
    questions = [q.strip('- ').strip() for q in questions_text.split('\n') if q.strip()]
    logging.debug(f"Extracted {len(questions)} improvement questions.")
    return questions

def extract_gaps(analysis):
    """Extract gaps from the analysis."""
    # Enhanced regex patterns for gaps
    patterns = [
        r"3\.\s*Gaps That Could Be Addressed by External Research\s*[:\-]?\s*(.*?)(4\.|$)",  # Primary pattern
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

        Please explain evaluate whether this paper might help address the identified questions and gaps. If specific questions or gaps are directly addressed, mention them. Otherwise, don't include it in the explanation.
        """
        
        # Get the response from LLM
        response = llm.invoke(prompt)
        explanation = response.content.strip()
        return explanation
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return "An error occurred while generating the explanation with the LLM."

def main():
    logging.info("=== Starting PDF Analysis Script ===")
    
    # Get the correct docs_path
    docs_path = get_project_paths()
    
    if not docs_path.exists():
        logging.error(f"Documents path does not exist: {docs_path}")
        return
        
    # Process each PDF in the folder
    pdf_files = list(docs_path.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        logging.info(f"\nAnalyzing {pdf_file.name}...")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            logging.warning(f"No text extracted from {pdf_file.name}")
            continue
        
        # Extract keywords
        logging.info("Extracting keywords...")
        keywords = extract_keywords(text)
        for term, freq in keywords.items():
            logging.info(f"Keyword: {term} (frequency: {freq})")
        
        # LLM Analysis
        logging.info("Starting document analysis...")
        analysis_results = analyze_document_with_llm(text)
        
        # Save analysis results as text
        output_file = docs_path / f"{pdf_file.stem}_analysis.txt"
        logging.info(f"Saving analysis results to {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("DOCUMENT ANALYSIS\n")
                f.write("================\n\n")
                f.write("KEY TERMS:\n")
                for term, freq in keywords.items():
                    f.write(f"- {term}: {freq}\n")
                f.write("\nANALYSIS AND QUESTIONS:\n")
                f.write(analysis_results["combined_analysis"])
                f.write("\n\nIMPROVEMENT QUESTIONS:\n")
                for question in analysis_results["improvement_questions"]:
                    f.write(f"- {question}\n")
                f.write("\nGAPS:\n")
                for gap in analysis_results["gaps"]:
                    f.write(f"- {gap}\n")
            logging.info("Analysis results saved successfully")
        except Exception as e:
            logging.error(f"Error saving analysis results: {e}")
        
        # Save retrieved papers
        recommendation_file = docs_path / f"{pdf_file.stem}_external-paper_recommendation.txt"
        logging.info(f"Saving retrieved papers to {recommendation_file}")
        try:
            with open(recommendation_file, 'w', encoding='utf-8') as f:
                f.write("EXTERNAL PAPER RECOMMENDATIONS\n")
                f.write("=============================\n\n")
                if analysis_results["retrieved_papers"]:
                    for paper in analysis_results["retrieved_papers"]:
                        # Use LLM to generate the explanation
                        explanation = explain_how_papers_help_with_llm(
                            paper_title=paper['title'], 
                            snippet=paper['snippet'], 
                            questions_and_gaps=analysis_results["improvement_questions"] + analysis_results["gaps"],
                            api_key=api_key
                        )
                        f.write(f"- {paper['title']}:\n")
                        f.write(f"  {explanation}\n\n")
                else:
                    f.write("No relevant papers found.\n")
            logging.info("Retrieved papers saved successfully")
        except Exception as e:
            logging.error(f"Error saving retrieved papers: {e}")

if __name__ == "__main__":
    main()