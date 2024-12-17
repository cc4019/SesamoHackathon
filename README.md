# SesamoHackathon -- Virtual Scholar Agent System

This repository contains a **multi-agent workflow** designed to function as a *virtual scholar agent*. The system automates summarizing team documents, analyzing research opportunities, retrieving relevant academic papers, and generating actionable insights for research teams.

## Overview
The **Virtual Scholar Agent System** operates on a daily, weekly, or monthly basis to streamline knowledge synthesis and research strategy for teams. 

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
