import os
from pathlib import Path
import PyPDF2
from collections import Counter
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging
import sys

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
    """Extract key terms using spaCy."""
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

def analyze_document_with_llm(text):
    """Use LLM to analyze document and generate questions."""
    logging.debug("Starting LLM analysis")
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4-turbo-preview", 
            temperature=0.7
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        logging.debug(f"Split text into {len(chunks)} chunks")
        
        prompt = """Analyze this document excerpt and provide:
        1. Key research areas and concepts
        2. Potential improvement questions
        3. Gaps that could be addressed by external research
        
        Document text:
        {text}
        """
        
        analysis = []
        for i, chunk in enumerate(chunks, 1):
            logging.debug(f"Processing chunk {i}/{len(chunks)}")
            try:
                response = llm.invoke(prompt.format(text=chunk))
                analysis.append(response.content)
            except Exception as e:
                logging.error(f"Error in LLM analysis of chunk {i}: {e}")
                logging.error("This might be due to API key issues or rate limiting")
        
        return "\n".join(analysis)
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        logging.error("This might be due to API key configuration issues")
        return "Error: Could not complete LLM analysis"

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
        analysis = analyze_document_with_llm(text)
        
        # Save results
        output_file = docs_path / f"{pdf_file.stem}_analysis.txt"
        logging.info(f"Saving results to {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("DOCUMENT ANALYSIS\n")
                f.write("================\n\n")
                f.write("KEY TERMS:\n")
                for term, freq in keywords.items():
                    f.write(f"- {term}: {freq}\n")
                f.write("\nANALYSIS AND QUESTIONS:\n")
                f.write(analysis)
            logging.info("Results saved successfully")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 