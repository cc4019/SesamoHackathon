from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
import os
import subprocess
import logging
from pathlib import Path
import zipfile
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import PyPDF2  # PyPDF2 for PDF text extraction
from chromadb import Client, Settings
import uuid
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()
# Initialize the LLM with contextual knowledge
api_key = os.getenv("OPENAI_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/team_docs'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize Chroma and Collections
chroma_client = Client(Settings(persist_directory="./chroma_data", anonymized_telemetry=False))
team_docs_collection_name = "team_docs"
external_papers_collection_name = "external_papers"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

team_docs_collection = chroma_client.get_or_create_collection(name=team_docs_collection_name)
external_papers_collection = chroma_client.get_or_create_collection(name=external_papers_collection_name)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def zip_files(output_filename, directories):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=directory)
                    zipf.write(file_path, arcname)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def load_contextual_knowledge(directories):
    knowledge = ""
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.pdf'):
                    knowledge += extract_text_from_pdf(file_path) + "\n"
                else:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            knowledge += f.read() + "\n"
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            knowledge += f.read() + "\n"
    return knowledge

def index_documents_in_chroma(collection, directories):
    """Indexes documents into Chroma collections."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.pdf'):
                    content = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    collection.add(
                        ids=[str(uuid.uuid4())],
                        documents=[chunk],
                        metadatas=[{"source": file_path}],
                        embeddings=[embedding_model.encode(chunk)]
                    )
    logging.info(f"Documents indexed in Chroma collection: {collection.name}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"File uploaded: {file_path}")
        
        return redirect(url_for('uploaded_file', filename=filename))
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    try:
        # Run the analyze_team_docs.py script
        subprocess.run(["python", "./scripts/analyze_team_docs.py"], check=True)
        logging.info("Analysis script executed successfully.")
        
        # Read the results from the file
        result_file_path = Path("./data/analysis_results/summarized_external-paper_recommendation.txt")
        with open(result_file_path, 'r', encoding='utf-8') as file:
            result_content = file.read()
        
        # Zip the files in data/analysis_results and data/external-paper
        zip_filename = "analysis_results.zip"
        directories_to_zip = [
            "./data/analysis_results",
            "./data/external-paper"
        ]
        zip_files(zip_filename, directories_to_zip)
        
        # Index documents in Chroma
        index_documents_in_chroma(team_docs_collection, [app.config['UPLOAD_FOLDER']])
        index_documents_in_chroma(external_papers_collection, directories_to_zip)
        
        return result_content
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running analysis script: {e}")
        return "Error starting analysis. Check the logs for details."

@app.route('/download_zip')
def download_zip():
    zip_path = Path("analysis_results.zip")
    if zip_path.exists():
        return send_from_directory(directory=str(zip_path.parent), path=zip_path.name, as_attachment=True)
    else:
        return "Zip file not found.", 404

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Generates chatbot responses based on indexed documents."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message received.'})

    try:
        # Retrieve context from team documents
        team_results = team_docs_collection.query(query_texts=[user_message], n_results=5)
        team_context = "\n".join(team_results['documents'][0])

        # Retrieve context from external papers
        external_results = external_papers_collection.query(query_texts=[user_message], n_results=5)
        external_context = "\n".join(team_results['documents'][0])

        # Combine contexts
        combined_context = f"Team Context:\n{team_context}\n\nExternal Analysis Context:\n{external_context}"

        # Generate response using LLM
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.7
        )
        prompt = f"""
        You are Sesemo, a virtual scholar advisor. Based on the following contexts, provide answer to the user query:

        {combined_context}

        User Query: {user_message}
        """
        response = llm.invoke(prompt)
        return jsonify({'response': response.content.strip()})
    except Exception as e:
        logging.error(f"Error in chatbot processing: {e}")
        return jsonify({'response': 'An error occurred while generating the response.'})

if __name__ == '__main__':
    app.run(debug=True)