from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
import os
import subprocess
import logging
from pathlib import Path
import zipfile
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Initialize the LLM with contextual knowledge
api_key = os.getenv("OPENAI_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/team_docs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

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

def load_contextual_knowledge(directories):
    knowledge = ""
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge += f.read() + "\n"
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        knowledge += f.read() + "\n"
    return knowledge

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
        
        # Load contextual knowledge
        global contextual_knowledge
        contextual_knowledge = load_contextual_knowledge(directories_to_zip)
        
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
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message received.'})
    
    # Initialize the LLM with contextual knowledge
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({'response': 'API key not found.'})
    
    # Initialize the LLM with contextual knowledge
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    prompt = f"""
    You are Sesemo, a virtual scholar advisor. Use the following contextual knowledge to answer the user's questions:
    
    {contextual_knowledge[:10000]}
    
    User: {user_message}
    Sesemo:
    """
    
    try:
        response = llm.invoke(prompt)
        return jsonify({'response': response.content.strip()})
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return jsonify({'response': 'An error occurred while generating the response.'})

if __name__ == '__main__':
    app.run(debug=True)