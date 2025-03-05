import json
import logging
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, TypedDict, Literal
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangSmith imports
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import traceable
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

# ArXiv imports
from arxiv import Client, Search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Paper-Retrieval-Workflow"
# Set these in your environment or add them here
# os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Load environment variables from .env file
load_dotenv()

# Check if the API key is available
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables. Make sure it's set in your .env file.")

# Initialize LangSmith client
try:
    langsmith_client = LangSmithClient()
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith client: {str(e)}")
    langsmith_client = None

# Pydantic models for data validation
class Paper(BaseModel):
    title: str = Field(..., description="Title of the paper")
    arxiv_id: str = Field(..., description="ArXiv ID of the paper", pattern=r'^\d{4}\.\d{5}(v\d+)?$')

class PaperEvaluation(BaseModel):
    title: str = Field(..., description="Title of the paper")
    relevance_points: List[str] = Field(..., description="List of relevance points", min_length=1)
    relevance_score: float = Field(..., description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)

class SearchQuery(BaseModel):
    topic: str = Field(..., description="Topic of the search query")
    query: str = Field(..., description="The search query string")

class RetrievalResults(BaseModel):
    retrieved_papers: List[Paper] = Field(default_factory=list, description="List of retrieved papers")
    paper_evaluations: Dict[str, dict] = Field(default_factory=dict, description="Paper evaluations by title")
    refined_search_queries: Dict[str, str] = Field(default_factory=dict, description="Refined search queries by topic")

# State definition for the graph
class GraphState(TypedDict):
    analysis_points: Dict[str, List[str]]
    context: str
    retrieved_papers: List[Dict]
    paper_evaluations: List[Dict]
    refined_queries: List[Dict]
    avg_score: float
    need_refinement: bool

# ArXiv search tool
class ArxivSearchTool(BaseTool):
    """Tool for searching ArXiv papers using the official API"""
    name: str = "search_arxiv"
    description: str = """
    Search ArXiv for academic papers. 
    Input should be a search query string.
    Returns a JSON object with a "papers" array containing paper details.
    """

    def __init__(self):
        super().__init__()
        logger.info("ArxivSearchTool initialized with direct API access")

    def _run(self, query: str, max_results: int = 10) -> str:
        """Execute the ArXiv search using the official API"""
        import requests
        import xml.etree.ElementTree as ET
        from datetime import datetime
        
        # Use the URL directly here instead of as a class attribute
        base_url = "http://export.arxiv.org/api/query"
        
        logger.info(f"ArxivSearchTool executing query: {query}")
        
        # Remove quotes for the actual API call
        clean_query = query.strip('"')
        
        # Prepare the API request
        params = {
            'search_query': clean_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        # Make the API request
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            error_msg = f"ArXiv API returned status code {response.status_code}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Parse the XML response
        root = ET.fromstring(response.content)
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        results = []
        
        # Extract entries (papers)
        entries = root.findall('.//atom:entry', namespaces)
        
        for entry in entries:
            try:
                # Extract paper details
                title = entry.find('./atom:title', namespaces).text.strip()
                
                # Get ID and convert to arxiv ID format
                id_url = entry.find('./atom:id', namespaces).text
                arxiv_id = id_url.split('/abs/')[-1]
                
                # Get summary/abstract
                summary = entry.find('./atom:summary', namespaces).text.strip()
                
                # Get authors
                author_elements = entry.findall('./atom:author/atom:name', namespaces)
                authors = [author.text for author in author_elements]
                
                # Get published date
                published_text = entry.find('./atom:published', namespaces).text
                try:
                    published_date = datetime.strptime(published_text, "%Y-%m-%dT%H:%M:%SZ")
                    published = published_date.strftime("%Y-%m-%d")
                except:
                    published = published_text
                
                paper = {
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "abstract": summary,
                    "authors": authors,
                    "published": published
                }
                
                results.append(paper)
                logger.info(f"Found paper: {paper['title']} ({paper['arxiv_id']})")
                
            except Exception as e:
                logger.warning(f"Error processing paper from API: {str(e)}")
                continue
        
        if not results:
            error_msg = f"No papers found for query: {query}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        json_results = json.dumps({"papers": results}, indent=2)
        logger.info(f"Returning {len(results)} papers")
        return json_results

@traceable(run_type="chain")
def load_analysis_results():
    """Load and process analysis results from JSON"""
    analysis_file = Path("scripts/data/analysis_results/analysis_results.json")
    
    try:
        with open(analysis_file, "r", encoding='utf-8') as f:
            results = json.load(f)
        
        analysis_points = {
            "research_gaps": [],
            "key_research_areas": [],
            "critical_analysis": [],
            "keywords": []
        }
        
        if isinstance(results, dict):
            for doc_name, analysis in results.items():
                if isinstance(analysis, dict):
                    analysis_points["research_gaps"].extend(analysis.get("research_gaps", []))
                    analysis_points["key_research_areas"].extend(analysis.get("key_research_areas", []))
                    analysis_points["critical_analysis"].extend(analysis.get("critical_analysis", []))
                    analysis_points["keywords"].extend(analysis.get("keywords", []))
        elif isinstance(results, list):
            analysis_points["research_gaps"].extend(results)
        
        for key in analysis_points:
            analysis_points[key] = list(dict.fromkeys(analysis_points[key]))
        
        if not any(analysis_points.values()):
            raise ValueError("No analysis points found in results")
        
        return analysis_points
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Analysis results file not found at {analysis_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {analysis_file}")

# Helper function to extract JSON from text
@traceable(run_type="chain")
def extract_json(text: str) -> dict:
    """Extract JSON from text output"""
    try:
        # Try direct JSON parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON object in text
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty dict
        logger.warning("Could not extract JSON from text")
        return {}

# Node functions
@traceable(run_type="chain")
def initialize(state: GraphState) -> GraphState:
    """Initialize the workflow state with analysis context"""
    analysis_points = state["analysis_points"]
    context = create_context_from_analysis(analysis_points)
    
    # Update state
    state["context"] = context
    logger.info("Initialized workflow with analysis context")
    
    return state

@traceable(run_type="agent")
def paper_search_agent(state: GraphState) -> GraphState:
    """Agent node for searching and retrieving academic papers"""
    context = state["context"]
    analysis_points = state["analysis_points"]
    
    # Step 1: Generate optimized search queries using LLM
    logger.info("Generating search queries with LLM")
    search_queries = generate_search_queries(context, analysis_points)
    
    # Step 2: Execute searches and collect papers
    logger.info(f"Executing {len(search_queries)} search queries")
    papers = execute_paper_searches(search_queries)
    
    # Step 3: Update state with retrieved papers
    # If we already have papers, append new ones without duplicates
    existing_papers = state.get("retrieved_papers", [])
    
    # Combine existing and new papers, avoiding duplicates
    combined_papers = existing_papers.copy()
    for paper in papers:
        if not any(p.get("arxiv_id") == paper.get("arxiv_id") for p in combined_papers):
            combined_papers.append(paper)
    
    state["retrieved_papers"] = combined_papers
    logger.info(f"Retrieved {len(papers)} new papers, total: {len(combined_papers)}")
    
    return state

@traceable(run_type="chain")
def evaluate_papers(state: GraphState) -> GraphState:
    """Node for evaluating paper relevance using parallel processing"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    context = state["context"]
    retrieved_papers = state.get("retrieved_papers", [])
    
    if not retrieved_papers:
        logger.warning("No papers to evaluate")
        state["paper_evaluations"] = []
        state["avg_score"] = 0.0
        state["need_refinement"] = True
        return state
    
    # Log each paper to be evaluated
    logger.info(f"Starting evaluation of {len(retrieved_papers)} papers with detailed criteria")
    for i, paper in enumerate(retrieved_papers):
        if isinstance(paper, dict) and "title" in paper:
            logger.info(f"Paper {i+1}: {paper['title']} (ID: {paper.get('arxiv_id', 'Unknown')})")
        else:
            logger.warning(f"Paper {i+1} has invalid format: {paper}")
    
    # Process papers in parallel with a thread pool
    all_evaluations = []
    max_workers = min(5, len(retrieved_papers))  # Reduce to 5 workers to avoid rate limits
    
    # Create a list to track which papers were processed
    processed_papers = []
    
    try:
        # Create a dictionary to map papers to their futures
        paper_futures = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers for evaluation
            for paper in retrieved_papers:
                if isinstance(paper, dict) and "title" in paper and "arxiv_id" in paper:
                    future = executor.submit(evaluate_single_paper, paper, context)
                    paper_futures[future] = paper
                    logger.info(f"Submitted paper for evaluation: {paper['title']} (ID: {paper['arxiv_id']})")
                else:
                    logger.warning(f"Skipping invalid paper: {paper}")
            
            # Wait for all futures to complete
            logger.info(f"Waiting for {len(paper_futures)} paper evaluations to complete...")
            
            # Use as_completed to process results as they finish
            for future in as_completed(paper_futures):
                paper = paper_futures[future]
                try:
                    result = future.result(timeout=180)  # 3 minute timeout
                    
                    # Ensure the result has the arxiv_id
                    if "arxiv_id" not in result and "arxiv_id" in paper:
                        result["arxiv_id"] = paper["arxiv_id"]
                        
                    all_evaluations.append(result)
                    processed_papers.append(paper.get('arxiv_id', 'Unknown'))
                    logger.info(f"Successfully evaluated paper: {paper.get('title', 'Unknown')} (ID: {paper.get('arxiv_id', 'Unknown')})")
                except Exception as e:
                    logger.error(f"Thread execution failed for paper {paper.get('title', 'Unknown')}: {str(e)}")
                    # Add a minimal evaluation to ensure we have something for each paper
                    all_evaluations.append({
                        "title": paper.get("title", "Unknown"),
                        "arxiv_id": paper.get("arxiv_id", "Unknown"),
                        "criteria_scores": {
                            "research_gap_alignment": 5,
                            "methodological_relevance": 5,
                            "theoretical_contribution": 5,
                            "practical_application": 5,
                            "innovation": 5
                        },
                        "overall_score": 0.5,
                        "relevance_points": ["Evaluation failed but paper might be relevant"],
                        "application_suggestions": ["Consider reviewing this paper manually"],
                        "strengths": ["Could not be automatically evaluated"],
                        "limitations": ["Automatic evaluation failed"]
                    })
        
        # Log completion of all evaluations
        logger.info(f"All paper evaluations completed. Processed {len(processed_papers)} papers.")
        
    except Exception as e:
        logger.error(f"Error in ThreadPoolExecutor: {str(e)}")
    
    # Log summary of processed papers
    logger.info(f"Processed {len(processed_papers)} out of {len(retrieved_papers)} papers")
    logger.info(f"Collected {len(all_evaluations)} evaluations")
    
    # Check if any papers were missed
    missing_papers = []
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "arxiv_id" in paper:
            if paper["arxiv_id"] not in processed_papers:
                missing_papers.append((paper["arxiv_id"], paper.get("title", "Unknown")))
    
    if missing_papers:
        logger.warning(f"Missing evaluations for {len(missing_papers)} papers: {missing_papers}")
        # Process missing papers sequentially as a fallback
        for paper in retrieved_papers:
            if isinstance(paper, dict) and "arxiv_id" in paper and paper["arxiv_id"] in [p[0] for p in missing_papers]:
                try:
                    logger.info(f"Processing missed paper sequentially: {paper['title']} (ID: {paper['arxiv_id']})")
                    result = evaluate_single_paper(paper, context)
                    
                    # Ensure the result has the arxiv_id
                    if "arxiv_id" not in result:
                        result["arxiv_id"] = paper["arxiv_id"]
                        
                    all_evaluations.append(result)
                    logger.info(f"Successfully evaluated missed paper: {paper['title']} (ID: {paper['arxiv_id']})")
                except Exception as e:
                    logger.error(f"Sequential evaluation failed for paper {paper['title']}: {str(e)}")
    
    # Process results
    try:
        # Update state
        state["paper_evaluations"] = all_evaluations
        
        # Calculate average score
        scores = [eval_data.get("overall_score", 0.0) for eval_data in all_evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        state["avg_score"] = avg_score
        state["need_refinement"] = avg_score < 0.6
        
        logger.info(f"Evaluated {len(all_evaluations)} papers with average score {avg_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error processing evaluation results: {str(e)}")
        state["paper_evaluations"] = []
        state["avg_score"] = 0.0
        state["need_refinement"] = True
    
    return state

@traceable(run_type="chain")
def format_and_save_results(state: GraphState) -> GraphState:
    """Format the final results, save them to file, and print a summary"""
    logger.info("Starting to format and save final results")
    
    # Create a new state dictionary to ensure we're not modifying the original
    new_state = dict(state)
    
    # Get the papers and evaluations
    retrieved_papers = new_state.get("retrieved_papers", [])
    paper_evaluations = new_state.get("paper_evaluations", [])
    
    # Log what we're working with
    logger.info(f"Formatting {len(retrieved_papers)} papers and {len(paper_evaluations)} evaluations")
    
    # Debug: Check if paper_evaluations have arxiv_ids
    arxiv_ids_present = sum(1 for eval_data in paper_evaluations if isinstance(eval_data, dict) and "arxiv_id" in eval_data)
    logger.info(f"Found {arxiv_ids_present} evaluations with arxiv_id out of {len(paper_evaluations)}")
    
    # Create a mapping from title to arxiv_id using retrieved_papers
    title_to_arxiv = {}
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "title" in paper and "arxiv_id" in paper:
            # Normalize the title by removing extra whitespace
            normalized_title = ' '.join(paper["title"].split())
            title_to_arxiv[normalized_title] = paper["arxiv_id"]
            # Also store with original title as a fallback
            title_to_arxiv[paper["title"]] = paper["arxiv_id"]
    
    # Ensure all evaluations have arxiv_ids
    for eval_data in paper_evaluations:
        if isinstance(eval_data, dict) and "title" in eval_data and "arxiv_id" not in eval_data:
            # Try to find the arxiv_id using the title
            title = eval_data["title"]
            normalized_title = ' '.join(title.split())
            
            if normalized_title in title_to_arxiv:
                eval_data["arxiv_id"] = title_to_arxiv[normalized_title]
                logger.info(f"Added arxiv_id to evaluation for '{title}' using normalized title")
            elif title in title_to_arxiv:
                eval_data["arxiv_id"] = title_to_arxiv[title]
                logger.info(f"Added arxiv_id to evaluation for '{title}' using exact title")
            else:
                # Try to find a close match
                for paper_title, arxiv_id in title_to_arxiv.items():
                    if title in paper_title or paper_title in title:
                        eval_data["arxiv_id"] = arxiv_id
                        logger.info(f"Added arxiv_id to evaluation for '{title}' using partial match")
                        break
    
    # Create evaluation dictionary for easier lookup using arxiv_id
    evaluation_dict = {}
    for eval_data in paper_evaluations:
        if isinstance(eval_data, dict) and "arxiv_id" in eval_data:
            evaluation_dict[eval_data["arxiv_id"]] = eval_data
    
    logger.info(f"Created evaluation lookup with {len(evaluation_dict)} entries by arxiv_id")
    
    # Filter papers by relevance score
    filtered_papers = []
    for paper in retrieved_papers:
        if not isinstance(paper, dict) or "arxiv_id" not in paper:
            continue
            
        arxiv_id = paper["arxiv_id"]
        if arxiv_id in evaluation_dict:
            score = evaluation_dict[arxiv_id].get("overall_score", 0.0)
            if score > 0.5:
                filtered_papers.append(paper)
                logger.info(f"Keeping paper '{paper.get('title', 'Unknown')}' (ID: {arxiv_id}) with score {score}")
            else:
                logger.info(f"Filtering out paper '{paper.get('title', 'Unknown')}' (ID: {arxiv_id}) with low score {score}")
        else:
            # If no evaluation exists, keep the paper
            filtered_papers.append(paper)
            logger.info(f"Keeping paper '{paper.get('title', 'Unknown')}' (ID: {arxiv_id}) with no evaluation")
    
    logger.info(f"Filtered papers from {len(retrieved_papers)} to {len(filtered_papers)} (relevance > 0.5)")
    
    # Create a formatted result structure
    formatted_results = {
        "retrieved_papers": filtered_papers,
        "paper_evaluations": {}
    }
    
    # Add evaluations to formatted_results
    for paper in filtered_papers:
        if isinstance(paper, dict) and "arxiv_id" in paper:
            arxiv_id = paper["arxiv_id"]
            title = paper.get("title", "Unknown")
            
            if arxiv_id in evaluation_dict:
                eval_data = evaluation_dict[arxiv_id]
                formatted_results["paper_evaluations"][title] = {
                    "arxiv_id": arxiv_id,
                    "relevance_points": eval_data.get("relevance_points", []),
                    "relevance_score": eval_data.get("overall_score", 0.0),
                    "application_suggestions": eval_data.get("application_suggestions", []),
                    "strengths": eval_data.get("strengths", []),
                    "limitations": eval_data.get("limitations", []),
                    "criteria_scores": eval_data.get("criteria_scores", {})
                }
    
    # Add formatted results to state
    new_state["formatted_results"] = formatted_results
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save results to file
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"paper_retrieval_results_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
        
        # Also save the research application summary
        summary_file = save_research_application_summary(formatted_results, output_dir)
        if summary_file:
            logger.info(f"Research application summary saved to {summary_file}")
            # Add the summary file path to the state
            new_state["summary_file"] = str(summary_file)
    except Exception as e:
        error_msg = f"Failed to save results to file: {str(e)}"
        logger.error(error_msg)
    
    # Return the updated state
    return new_state

def create_context_from_analysis(analysis_points):
    """Create context string from analysis points"""
    context_sections = []
    if analysis_points["research_gaps"]:
        context_sections.append("Research Gaps:\n" + "\n".join([f"- {gap}" for gap in analysis_points["research_gaps"]]))
    if analysis_points["key_research_areas"]:
        context_sections.append("Key Research Areas:\n" + "\n".join([f"- {area}" for area in analysis_points["key_research_areas"]]))
    if analysis_points["critical_analysis"]:
        context_sections.append("Critical Analysis Points:\n" + "\n".join([f"- {point}" for point in analysis_points["critical_analysis"]]))
    
    return "\n\n".join(context_sections)

def generate_search_queries(context: str, analysis_points: Dict) -> List[str]:
    """Generate optimized ArXiv search keywords using existing keywords and LLM"""
    # First, check if we have keywords in the analysis_points
    if analysis_points.get("keywords") and len(analysis_points["keywords"]) >= 3:
        # We have enough keywords, use them directly
        keywords = analysis_points["keywords"][:5]  # Take up to 5 keywords
        
        # Format keywords for ArXiv search
        formatted_keywords = []
        for keyword in keywords:
            # Clean up the keyword
            clean_keyword = keyword.strip()
            # Keep keywords short and focused
            if len(clean_keyword.split()) > 3:
                # For longer keywords, take just the key terms
                words = clean_keyword.split()
                clean_keyword = " ".join(words[:3])
            # Add quotes for exact matching
            formatted_keywords.append(f'"{clean_keyword}"')
        
        logger.info(f"Using keywords from analysis_results: {formatted_keywords[:3]}")
        return formatted_keywords  # Return top 3 keywords

def execute_paper_searches(search_queries: List[str]) -> List[Dict]:
    """Execute paper searches using the provided queries"""
    # Create ArXiv search tool
    arxiv_tool = ArxivSearchTool()
    
    # Execute searches and collect papers
    papers = []
    
    # First, try searching with all keywords combined (AND search)
    if len(search_queries) >= 3:
        try:
            # Combine the first 3 keywords with AND operator
            # Remove quotes for combining
            clean_queries = [q.strip('"') for q in search_queries[:3]]
            combined_query = f'"{clean_queries[0]} AND {clean_queries[1]} AND {clean_queries[2]}"'
            
            logger.info(f"Searching with combined keywords: {combined_query}")
            search_results = json.loads(arxiv_tool._run(combined_query))
            
            if "papers" in search_results and search_results["papers"]:
                for paper in search_results["papers"]:
                    # Add to papers if not already there
                    if not any(p.get("arxiv_id") == paper.get("arxiv_id") for p in papers):
                        papers.append({
                            "title": paper.get("title", ""),
                            "arxiv_id": paper.get("arxiv_id", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": paper.get("authors", []),
                            "published": paper.get("published", "")
                        })
                logger.info(f"Found {len(search_results['papers'])} papers with combined search")
        except Exception as e:
            logger.warning(f"Combined search failed: {str(e)}")
    
    # If combined search didn't yield enough results, try individual keywords
    if len(papers) < 5:
        for query in search_queries:
            try:
                logger.info(f"Searching for individual keyword: {query}")
                search_results = json.loads(arxiv_tool._run(query))
                if "papers" in search_results:
                    for paper in search_results["papers"]:
                        # Add to papers if not already there
                        if not any(p.get("arxiv_id") == paper.get("arxiv_id") for p in papers):
                            papers.append({
                                "title": paper.get("title", ""),
                                "arxiv_id": paper.get("arxiv_id", ""),
                                "abstract": paper.get("abstract", ""),
                                "authors": paper.get("authors", []),
                                "published": paper.get("published", "")
                            })
            except Exception as e:
                logger.error(f"Error searching for {query}: {str(e)}")
    
    # If we still don't have enough papers, try pairs of keywords
    if len(papers) < 5 and len(search_queries) >= 2:
        for i in range(len(search_queries) - 1):
            for j in range(i + 1, len(search_queries)):
                try:
                    # Combine two keywords with AND
                    clean_query1 = search_queries[i].strip('"')
                    clean_query2 = search_queries[j].strip('"')
                    pair_query = f'"{clean_query1} AND {clean_query2}"'
                    
                    logger.info(f"Searching with keyword pair: {pair_query}")
                    search_results = json.loads(arxiv_tool._run(pair_query))
                    
                    if "papers" in search_results:
                        for paper in search_results["papers"]:
                            # Add to papers if not already there
                            if not any(p.get("arxiv_id") == paper.get("arxiv_id") for p in papers):
                                papers.append({
                                    "title": paper.get("title", ""),
                                    "arxiv_id": paper.get("arxiv_id", ""),
                                    "abstract": paper.get("abstract", ""),
                                    "authors": paper.get("authors", []),
                                    "published": paper.get("published", "")
                                })
                except Exception as e:
                    logger.warning(f"Pair search failed: {str(e)}")
    
    logger.info(f"Total unique papers found: {len(papers)}")
    return papers

@traceable(run_type="chain")
def retrieve_papers(state: GraphState) -> GraphState:
    """Node for retrieving papers using the paper search agent"""
    # Simply call the paper search agent
    return paper_search_agent(state)

@traceable(run_type="chain")
def save_research_application_summary(results, output_dir):
    """Save a concise summary of how each paper can help with the research"""
    # Create a simplified summary structure
    summary = {
        "summary_date": datetime.now().strftime("%Y-%m-%d"),
        "papers": []
    }
    
    # Extract relevant papers and their application suggestions
    retrieved_papers = results.get("retrieved_papers", [])
    paper_evaluations = results.get("paper_evaluations", {})
    
    # Create a lookup dictionary for paper evaluations by arxiv_id
    evaluations_by_id = {}
    
    # First, create a mapping from title to arxiv_id using retrieved_papers
    title_to_arxiv = {}
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "title" in paper and "arxiv_id" in paper:
            title_to_arxiv[paper["title"]] = paper["arxiv_id"]
    
    # Now create evaluations_by_id using both the arxiv_id in eval_data and the title mapping
    for title, eval_data in paper_evaluations.items():
        if isinstance(eval_data, dict):
            # If eval_data already has arxiv_id, use it
            if "arxiv_id" in eval_data:
                arxiv_id = eval_data["arxiv_id"]
                evaluations_by_id[arxiv_id] = eval_data
            # Otherwise, try to get arxiv_id from title_to_arxiv mapping
            elif title in title_to_arxiv:
                arxiv_id = title_to_arxiv[title]
                # Add arxiv_id to eval_data for future reference
                eval_data["arxiv_id"] = arxiv_id
                evaluations_by_id[arxiv_id] = eval_data
    
    logger.info(f"Created evaluation lookup with {len(evaluations_by_id)} entries by arxiv_id")
    
    for paper in retrieved_papers:
        print('Paper Title', paper['title'], '(ID:', paper.get('arxiv_id', 'Unknown'), ')')
        if not isinstance(paper, dict) or "arxiv_id" not in paper:
            continue
            
        title = paper.get("title", "Unknown Title")
        arxiv_id = paper.get("arxiv_id", "Unknown ID")
        
        # Skip papers without evaluations - check with arxiv_id
        if arxiv_id not in evaluations_by_id:
            logger.info(f"Skipping paper '{title}' - no evaluation data (ID: {arxiv_id})")
            continue
            
        # Get evaluation data using arxiv_id
        eval_data = evaluations_by_id[arxiv_id]
        overall_score = eval_data.get("overall_score", eval_data.get("relevance_score", 0.0))
        
        # Skip papers with relevance score <= 0.5
        if overall_score <= 0.5:
            logger.info(f"Skipping paper '{title}' - low relevance score: {overall_score}")
            continue
        
        # Create paper summary
        paper_summary = {
            "title": title,
            "arxiv_id": arxiv_id,
            "abstract": paper.get("abstract", "No abstract available")[:500] + "..." if len(paper.get("abstract", "")) > 500 else paper.get("abstract", "No abstract available"),
            "overall_score": overall_score,
            "criteria_scores": eval_data.get("criteria_scores", {}),
            "research_applications": eval_data.get("application_suggestions", []),
            "strengths": eval_data.get("strengths", []),
            "limitations": eval_data.get("limitations", [])
        }
        
        # Add top relevance points
        if "relevance_points" in eval_data and eval_data["relevance_points"]:
            paper_summary["key_relevance_points"] = eval_data["relevance_points"][:3]
        
        summary["papers"].append(paper_summary)
    
    # Sort papers by overall score (highest first)
    summary["papers"] = sorted(summary["papers"], key=lambda p: p.get("overall_score", 0), reverse=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"research_application_summary_{timestamp}.json"
    
    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Research application summary saved with {len(summary['papers'])} relevant papers (with detailed criteria)")
        return output_file
    except Exception as e:
        error_msg = f"Failed to save research application summary: {str(e)}"
        logger.error(error_msg)
        return None

@traceable(run_type="chain")
def evaluate_single_paper(paper, context):
    """Evaluate a single paper with more detailed scoring criteria"""
    try:
        # Create evaluation prompt template with more detailed scoring criteria
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are a research evaluator assessing the relevance of an academic paper.
        
        Analyze this paper considering the following research context:
        
        {context}
        
        Paper to evaluate:
        {paper}
        
        Evaluate on these specific criteria:
        1. Research Gap Alignment (0-10): How well it addresses the specific research gaps
        2. Methodological Relevance (0-10): How useful its methods are to the research
        3. Theoretical Contribution (0-10): How it contributes to theoretical understanding
        4. Practical Application (0-10): How applicable its findings are to practical problems
        5. Innovation (0-10): How novel or innovative the approach is
        
        Also provide:
        - At least 3 specific points about how the paper is relevant
        - 2-3 specific ways this paper can help with the existing research
        
        Provide a JSON response with:
        - "title": Title of the paper
        - "arxiv_id": ArXiv ID of the paper
        - "criteria_scores": Object with the 5 criteria scores
        - "overall_score": The average of all criteria scores, normalized to 0.0-1.0
        - "relevance_points": List of at least 3 points about how the paper is relevant
        - "application_suggestions": List of 2-3 specific ways this paper can help with the existing research
        - "strengths": List of 2-3 strengths of this paper
        - "limitations": List of 1-2 limitations or weaknesses of this paper
        """)
        
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Create evaluation chain
        evaluation_chain = evaluation_prompt | llm | StrOutputParser() | extract_json
        
        # Format paper for prompt
        paper_text = f"Title: {paper['title']}\nArXiv ID: {paper['arxiv_id']}\nAbstract: {paper.get('abstract', 'No abstract available')}"
        
        # Execute evaluation
        result = evaluation_chain.invoke({
            "context": context,
            "paper": paper_text
        })
        
        # Ensure we have all expected fields
        if "arxiv_id" not in result:
            result["arxiv_id"] = paper.get("arxiv_id", "Unknown")
            
        if "criteria_scores" not in result:
            result["criteria_scores"] = {
                "research_gap_alignment": 0,
                "methodological_relevance": 0,
                "theoretical_contribution": 0,
                "practical_application": 0,
                "innovation": 0
            }
        
        if "overall_score" not in result:
            # Calculate overall score from criteria if missing
            scores = result["criteria_scores"].values()
            result["overall_score"] = sum(scores) / (len(scores) * 10) if scores else 0.0
            
        if "strengths" not in result:
            result["strengths"] = []
            
        if "limitations" not in result:
            result["limitations"] = []
        
        logger.info(f"Evaluated paper: {paper['title']} (ID: {paper['arxiv_id']}) with score {result['overall_score']:.2f}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating paper {paper.get('title', 'Unknown')}: {str(e)}")
        # Return a minimal valid structure
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": paper.get("arxiv_id", "Unknown"),
            "criteria_scores": {
                "research_gap_alignment": 0,
                "methodological_relevance": 0,
                "theoretical_contribution": 0,
                "practical_application": 0,
                "innovation": 0
            },
            "overall_score": 0.0,
            "relevance_points": ["Evaluation failed"],
            "application_suggestions": ["Evaluation failed"],
            "strengths": [],
            "limitations": []
        }

@traceable(run_type="chain")
def main():
    """Main execution function"""
    try:
        # Load analysis results
        analysis_points = load_analysis_results()
        
        # Create workflow graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("initialize", initialize)
        workflow.add_node("retrieve", paper_search_agent)
        workflow.add_node("evaluate", evaluate_papers)
        workflow.add_node("format_and_save", format_and_save_results)
        
        # Add edges
        workflow.add_edge("initialize", "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_edge("evaluate", "format_and_save")
        workflow.add_edge("format_and_save", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Compile graph
        app = workflow.compile()
        
        # Initialize state
        initial_state = {
            "analysis_points": analysis_points,
            "context": "",
            "retrieved_papers": [],
            "paper_evaluations": [],
            "refined_queries": [],
            "avg_score": 0.0,
            "need_refinement": False
        }
        
        # Run workflow
        logger.info("Starting paper retrieval workflow")
        final_state = app.invoke(initial_state)
        
        # Debug log the final state
        logger.info(f"Final state keys: {list(final_state.keys())}")
        
        # Wait for all traces to be uploaded
        wait_for_all_tracers()
        
        return final_state.get("formatted_results", {})
        
    except Exception as e:
        logger.exception("Critical error occurred:")
        raise RuntimeError(f"Execution failed: {str(e)}") from e

if __name__ == "__main__":
    main()