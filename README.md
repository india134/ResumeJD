AI powered Resume–Job Description Matcher

Overview
The Intelligent Resume–JD Matcher is an advanced AI tool that semantically analyzes a candidate’s resume against a job description. Unlike simple keyword checkers, it uses multiple transformer models (SBERT, CodeBERT, GraphCodeBERT), an ontology-driven skill parser, and regex-based extraction to identify skills, evaluate semantic context, and generate detailed AI reports for both candidates and HR.
The system is built with Flask (backend) and Bootstrap (frontend), supports PDF parsing with OCR fallback, and delivers actionable insights including multi-criteria match scores, skill-gap analysis, and tailored AI reports.
________________________________________
Features
•	Advanced Resume Parsing
• Extracts text from resumes (PDF), including OCR for image-based files (PyMuPDF + Tesseract).
• Cleans and normalizes text for consistent downstream analysis.
•	Ontology-Driven Skill Extraction
• Uses a central skills_ontology. json with canonical skills + aliases.
• Matches skills in resumes and JDs with high precision using regex + ontology mapping.
•	Transformer-Based Semantic Scoring
• Embeds resume sections and JD skills using SBERT, CodeBERT, GraphCodeBERT.
• Computes similarity for Skills, Projects, Experience, and Qualifications individually.
• Generates an overall score by averaging multiple embedding model outputs.
•	AI-Generated Reports (Dual Persona)
• Candidate view: constructive feedback on strengths and missing skills.
• HR view: analytical summary of candidate fit, gaps, and evidence of experience.
• Reports include project suggestions for upskilling.
•	Dashboard Interface
• Upload resume PDF + paste JD text.
• Displays structured scores, missing skills, and interactive AI-generated feedback.
________________________________________
System Architecture
(User – Candidate / HR) → Flask Web App → Resume & JD Input → Text Extraction & Ontology Matching → Embeddings (SBERT, CodeBERT, GraphCodeBERT) → Scoring & Analysis → GPT-based Report Generator → Flask → Bootstrap Dashboard → User
________________________________________
Tech Stack
•	Backend: Flask
•	NLP/Embeddings: SBERT, CodeBERT, GraphCodeBERT (HuggingFace)
•	ML: PyTorch, Sentence-Transformers, Scikit-learn
•	OCR/Text Extraction: PyMuPDF, Tesseract
•	Generative AI: OpenAI GPT (dual persona reports)
•	Frontend: Bootstrap, HTML, CSS
________________________________________
Project Structure
resume-jd-matcher/
│── app.py: Main Flask app with scoring pipeline
│── skills_ontology. json: Skill ontology knowledge base
│── templates/: HTML dashboard (Bootstrap)
│── static/: CSS/JS files
│── requirements.txt: Dependencies
________________________________________

Challenges & Learnings
1.	Inaccurate Skill Extraction
o	Problem: Early keyword matching was brittle and flagged false positives (e.g., letter “R” mistaken for R language).
o	Solution: Designed ontology-driven system with aliases + regex word boundaries for precision.
2.	Semantic Similarity Issues
o	Problem: Embeddings sometimes treated unrelated skills as substitutes (e.g., AWS ≠ Docker).
o	Solution: Refined ontology categories and sibling-skill logic for nuanced similarity.
3.	Diverse Resume Formats
o	Problem: Couldn’t reliably detect resume sections with inconsistent headings.
o	Solution: Built synonym-based flexible section extractor (e.g., “Education” vs “Academic Background”).
4.	Performance Bottlenecks
o	Problem: Running 3 transformer models on CPU was too slow.
o	Solution: Enabled GPU acceleration in PyTorch for embeddings.
5.	Unprofessional AI Reports
o	Problem: Early GPT outputs were generic and poorly formatted.
o	Solution: Engineered structured prompts + persona-based templates (Candidate / HR) and rendered reports as clean HTML.
________________________________________
Author
Akash Gupta
M.Tech Artificial Intelligence, IIT Kharagpur

