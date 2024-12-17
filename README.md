# SesamoHackathon -- Virtual Scholar Agent System

This repository contains a **multi-agent workflow** designed to function as a *virtual scholar agent*. The system automates summarizing team documents, analyzing research opportunities, retrieving relevant academic papers, and generating actionable insights for research teams.

## Overview
The **Virtual Scholar Agent System** operates on a daily, weekly, or monthly basis to streamline knowledge synthesis and research strategy for teams. 
<img width="922" alt="image" src="https://github.com/user-attachments/assets/17eb03e8-7821-4cf1-b22a-849f313265bd" />

### Key Workflow Steps

1. **Centralized Upload Step**
   - Internal team documents are uploaded to a centralized folder.
   - Documents must be well-organized and tagged with metadata (e.g., project name, date).

2. **Project Context Agent**
   - Summarizes internal team documents under predefined categories:
     - **Research Framework & Key Concepts**
     - **Critical Analysis & Improvement Areas**
     - **Research Gaps & Extension Opportunities**
     - **Literature Review Strategy**: Keywords for paper search
   - Highlights **new information** compared to the previous week’s documents.

3. **Summary Project Context Agent**
   - Aggregates individual summaries while avoiding redundant insights.
   - Generates a set of optimized, refined **keywords** for paper retrieval.

4. **Rough Paper Retrieval Agent**
   - Fetches the latest academic papers (e.g., published in the past month) using the refined keywords.
   - Includes a **scoring mechanism** for relevance, based on metrics such as:
     - Citation count
     - Abstract similarity

5. **Granular Paper Retrieval Agent**
   - Matches paper embeddings against internal team documents for deeper **context alignment**.
   - Suggests tiered relevance levels:
     - **Highly Relevant**
     - **Moderately Relevant**

6. **Feedback Agent**
   - Annotates retrieved papers with **actionable insights** based on the final summaries.
   - Recommends **follow-up actions**, such as:
     - Suggested reading order
     - Collaboration opportunities

7. **Final Output**
   - A clear, categorized report that includes:
     - Summarized internal context
     - Highlighted research gaps and improvement areas
     - Refined keywords for literature search
     - Annotated external papers with actionable insights and tiered relevance levels
    
### Agents in Each Process
<img width="870" alt="image" src="https://github.com/user-attachments/assets/f857e036-01e6-4213-8a5d-1b168d8ffe9e" />

---

## Features
- **Automated Document Analysis**: Extracts and summarizes key research points from internal documents.
- **Change Detection**: Highlights new insights compared to previous document versions.
- **Keyword Optimization**: Generates refined keywords for precise literature search.
- **Relevance Scoring**: Assesses external papers based on citation count and content similarity.
- **Contextual Paper Filtering**: Embedding-based matching for deeper alignment.
- **Actionable Insights**: Annotates external papers with follow-up recommendations.
- **Flexible Scheduling**: Runs daily, weekly, or monthly depending on team needs.

## Workflow Diagram
```
[Input] Internal Documents (Centralized Folder)
   |
   v
[Project Context Agent] ---> [Summary Project Context Agent]
   |                                  |
   v                                  v
[Rough Paper Retrieval Agent]      [Summarized Research Context]
   |                                  |
   v                                  v
[Granular Paper Retrieval Agent] ---> [Feedback Agent]
   |
   v
[Final Output: Summarized Context + Annotated Papers + Actionable Insights]
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/virtual-scholar-agent.git
   cd virtual-scholar-agent
   ```
2. **Install Dependencies**:
   Ensure you have Python installed (>=3.8). Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Centralized Folder**:
   Create a directory for internal documents and configure it in the system's settings.

4. **Configure API Keys**:
   - For paper retrieval, integrate APIs such as Semantic Scholar, arXiv, or other academic databases.
   - Update `config.json` with relevant API keys.

## Usage
Run the main workflow script:
```bash
python main.py
```

The system will:
- Summarize internal documents.
- Retrieve relevant papers.
- Generate research insights and feedback.

Results will be saved in the `output/` folder.

## Outputs
The system generates a structured report including:
1. **Summarized Internal Context**
2. **Highlighted Research Gaps & Opportunities**
3. **Optimized Keywords for Literature Search**
4. **Annotated External Papers**
   - Tiered relevance levels
   - Follow-up actions and collaboration recommendations

---

## Future Enhancements
### Internal Artifacts Review
- Enable **multimodal data ingestion** to analyze various types of data, including:
   - Text documents (current implementation)
   - Images and diagrams (e.g., flowcharts, annotated graphs)
   - Audio and video content (e.g., meeting recordings, presentations)
- Generate insights that combine analysis across multiple data modalities for richer understanding.

### External Paper Retrieval
- **Scale search to multiple academic repositories and websites**:
   - Add support for databases such as IEEE Xplore, Springer, PubMed, and others.
   - Implement aggregation mechanisms to prioritize search results across platforms.
- Introduce **adaptive search strategies** to optimize queries dynamically based on project needs.

### Project Improvement Recommendation
- Enhance recommendation capabilities by:
   - Suggesting **proactive collaboration opportunities** with authors of relevant external papers.
   - Incorporating **trend analysis** to predict upcoming research directions and topics based on retrieved papers.
   - Providing **visualizations** of research gaps, collaboration pathways, and key insights for easier interpretation.

---

## Contributions
We welcome contributions! Please open an issue to suggest improvements or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or collaborations, reach out via email: **ivanye13671@gmail.com**, **chuciche@gmail.com**.

---

Happy researching! 🚀

