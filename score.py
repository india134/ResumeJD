import os
import io
import re
import json
import fitz
import torch
import openai
import numpy as np
import random
from PIL import Image
from flask import Flask, request, render_template
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from markdown import markdown

# =========================
# CONFIG & CONSTANTS
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

SEED = 42
PARTIAL_CREDIT_WEIGHT = 0.6
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = Flask(__name__)
app.secret_key = "supersecret"
client = openai.OpenAI(api_key=api_key)

# --- MODEL CONSTANTS RESTORED ---
MODEL_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CODEBERT = "microsoft/codebert-base"
MODEL_GRAPHCODEBERT = "microsoft/graphcodebert-base"
ONTOLOGY_PATH = "skills_ontology.json"


# =========================
# HELPER FUNCTIONS
# =========================
# In app.py, under the other constants
SECTION_SYNONYMS = {
    'projects': ['projects', 'personal projects', 'academic projects', 'portfolio'],
    'experience': ['experience', 'work experience', 'professional experience', 'internships', 'employment history'],
    'education': ['education', 'academic background', 'qualifications', 'academics'],
    'skills': ['skills', 'technical skills', 'proficiencies', 'technical expertise',"professional skills"]
}

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return re.sub(r'[^\x00-\x7F]+', ' ', text)
def extract_section_text(resume_text, section_key, synonyms_dict):
    """
    Intelligently extracts text for a given section using a dictionary of synonyms.
    """
    # Create a regex pattern from the list of synonyms for the target section
    section_patterns = '|'.join(synonyms_dict[section_key])
    
    # Create a list of all other section headings to use as "stop" words
    stop_patterns = []
    for key, values in synonyms_dict.items():
        if key != section_key:
            stop_patterns.extend(values)
    
    stop_regex = '|'.join(stop_patterns)
    
    # The final regex looks for a section heading and captures everything until the next heading or the end of the text
    pattern = re.compile(rf"({section_patterns})(.*?)(?=\n({stop_regex})|$)", re.IGNORECASE | re.DOTALL)
    
    match = pattern.search(resume_text)
    
    return match.group(2).strip() if match else ""
def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            else:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img) + "\n"
    return clean_text(text)

def load_ontology(path=ONTOLOGY_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_skill_maps(ontology):
    all_skills_and_aliases = set()
    for key, values in ontology.items():
        all_skills_and_aliases.add(key.lower())
        for value in values:
            all_skills_and_aliases.add(value.lower())
    return list(all_skills_and_aliases)

def extract_skills(text, all_skills_list):
    found_skills = set()
    for skill in all_skills_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.add(skill)
    return list(found_skills)

# --- MULTI-MODEL EMBEDDING FUNCTION RESTORED ---
def embed_text(texts, model_name):
    """Generates a vector embedding, handling both SentenceTransformer and base Transformer models."""
    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True)
    else:
        # This handles CodeBERT and GraphCodeBERT
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get a single vector
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Ensure the output is always a 2D array for cosine_similarity
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb

def get_similarity(a, b):
    score = cosine_similarity(a, b)[0][0]
    return (score + 1) / 2 # Normalize to 0-1



# =========================
# MAIN FLASK ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def dashboard():
    report_html, scores, details = None, None, None

    if request.method == "POST":
        # 1. Initial Setup: Get data from form and parse resume
        user_type = request.form.get("user_type", "candidate")
        jd_text = clean_text(request.form.get("jd_text", ""))
        resume_file = request.files.get("resume")
        if not resume_file:
            return "No resume file provided.", 400

        resume_data = io.BytesIO(resume_file.read())
        resume_text = extract_text_from_pdf(resume_data)
        
        ontology = load_ontology()
        all_skills_list = create_skill_maps(ontology)
        
        jd_skills = sorted(list(set(extract_skills(jd_text, all_skills_list))))
        resume_skills_set = set(extract_skills(resume_text, all_skills_list))

        # 2. Tiered Skill Matching for Overall Resume
        exact_matches, sibling_matches, missing_skills = [], {}, []
        inverted_ontology = {alias: {v.lower() for v in values} | {key.lower()} for key, values in ontology.items() for alias in [key.lower()] + [v.lower() for v in values]}

        for skill in jd_skills:
            if skill in resume_skills_set:
                exact_matches.append(skill)
            else:
                sibling_skills = inverted_ontology.get(skill, set())
                found_siblings = resume_skills_set.intersection(sibling_skills)
                if found_siblings:
                    sibling_matches[skill] = list(found_siblings)[0]
                else:
                    missing_skills.append(skill)
        
        # 3. Calculate Scores
        # A. Tiered Skill Score
        skills_score = 1.0 if not jd_skills else ((len(exact_matches) + len(sibling_matches) * PARTIAL_CREDIT_WEIGHT) / len(jd_skills))

        # B. Semantic Scores for Resume Sections
        projects_text = extract_section_text(resume_text, 'projects', SECTION_SYNONYMS)
        experience_text = extract_section_text(resume_text, 'experience', SECTION_SYNONYMS)
        qualifications_text = extract_section_text(resume_text, 'education', SECTION_SYNONYMS)
        
        models = [MODEL_SBERT, MODEL_CODEBERT, MODEL_GRAPHCODEBERT]
        jd_skills_text = " ".join(jd_skills)

        def calculate_average_score(section_text, target_text):
            if not section_text or not target_text: return 0.0
            model_scores = []
            for model in models:
                emb_section = embed_text([section_text], model)
                emb_target = embed_text([target_text], model)
                model_scores.append(get_similarity(emb_section, emb_target))
            return np.mean(model_scores) if model_scores else 0.0

        projects_score = calculate_average_score(projects_text, jd_skills_text)
        experience_score = calculate_average_score(experience_text, jd_skills_text)
        qualifications_keywords = "bachelor b.tech b.e bsc master m.s msc m.tech phd degree education university college institute"
        qualifications_score = calculate_average_score(qualifications_text, qualifications_keywords)
        
        # C. Final Weighted Overall Score
        weights = {"skills": 0.4, "projects": 0.3, "experience": 0.2, "qualifications": 0.1}
        overall = (skills_score * weights["skills"] + 
                   projects_score * weights["projects"] + 
                   experience_score * weights["experience"] + 
                   qualifications_score * weights["qualifications"])
        
        # 4. Prepare Detailed Analysis Lists for UI and Report
        project_skills_set = set(extract_skills(projects_text, all_skills_list))
        experience_skills_set = set(extract_skills(experience_text, all_skills_list))
        
        demonstrated_project_skills = sorted(list(set(jd_skills).intersection(project_skills_set)))
        demonstrated_experience_skills = sorted(list(set(jd_skills).intersection(experience_skills_set)))
        
        # Flexible "Missing Project Skills" logic
        missing_project_skills_list = []
        for skill in jd_skills:
            if skill not in project_skills_set:
                sibling_skills = inverted_ontology.get(skill, set())
                if not project_skills_set.intersection(sibling_skills):
                    missing_project_skills_list.append(skill)
        missing_project_skills = sorted(missing_project_skills_list)

        # 5. Assemble Final Data for Rendering
        scores = {"skills": skills_score, "projects": projects_score, "experience": experience_score, "qualifications": qualifications_score, "overall": overall}
        details = {
            "missing_skills": sorted(missing_skills), 
            "missing_project_skills": missing_project_skills,
            "demonstrated_project_skills": demonstrated_project_skills,
            "demonstrated_experience_skills": demonstrated_experience_skills
        }
        
        report_html = generate_report(
            user_type, 
            jd_skills, 
            list(resume_skills_set), 
            exact_matches, 
            sibling_matches, 
            missing_skills, 
            missing_project_skills,
            demonstrated_project_skills,
            demonstrated_experience_skills
        )

    return render_template("dashboard.html", scores=scores, details=details, report=report_html)

def generate_report(user_type, jd_skills, resume_skills, exact_matches, sibling_matches, missing_skills, missing_project_skills, demonstrated_project_skills, demonstrated_experience_skills):
    # ... (This function is correct and remains unchanged)
    base_info = f"""
    - Job Requirements (from JD): {", ".join(jd_skills)}
    - Candidate's Overall Skills (from Resume): {", ".join(resume_skills)}
    - Directly Matched Skills: {", ".join(exact_matches)}
    - Strongly Related Skills: {", ".join([f'{jd_skill} (covered by {resume_skill})' for jd_skill, resume_skill in sibling_matches.items()])}
    - Overall Skill Gaps: {", ".join(missing_skills)}
    - Skills Not Demonstrated in Projects: {", ".join(missing_project_skills)}
    - Skills Demonstrated in Work Experience: {", ".join(demonstrated_experience_skills)}
    """
    if user_type == 'candidate':
        prompt = f"""
        As an expert Career Coach, provide direct, constructive feedback to a candidate comparing their resume to a job description (JD).

        ### Candidate Skill Analysis
        {base_info}

        ### Your Feedback Instructions:
        1.  **Strengths for the Role:** Highlight the 'Directly Matched' and 'Strongly Related' skills. Explain how these make you a strong candidate for this specific job.
        2.  **Bridging the Gaps:** Discuss the 'Overall Skill Gaps'. For each, explain why it's important for this role and suggest a project to learn it.
        3.  **Enhancing Your Projects:** Look at the 'Skills Not Demonstrated in Projects'. Advise on how you could update your projects to explicitly showcase these valuable skills.
        4.  **Work Experience Highlights:** Analyze the 'Skills Demonstrated in Work Experience'. Explain how proving these skills in a professional setting makes your profile much stronger for this role.
        ### Formatting Rules:
        - Use a direct, encouraging, and expert tone. Use clean headings.
        - DO NOT use a letter format or markdown asterisks (`**`) and write like you are providing feedback to a candidate and referring to canddiate only
        """
    else: # user_type == 'hr'
        prompt = f"""
        As an expert Technical Analyst, provide an objective analysis of a candidate's resume against a job description (JD) for a hiring manager.

        ### Candidate-JD Alignment Report
        {base_info}

        ### Your Analysis Instructions:
        1.  **Executive Summary:** Provide a brief, top-level summary of the candidate's fit for the role.
        2.  **Alignment Areas:** Detail the 'Directly Matched' and 'Strongly Related' skills and analyze how they meet the core job requirements.
        3.  **Identified Gaps:** List the 'Overall Skill Gaps' and assess their criticality for the role.
        4.  **Project Experience Insight:** Analyze the 'Skills Not Demonstrated in Projects'. Note that while the candidate lists these skills, their absence from project descriptions may indicate a lack of practical experience, a potential area to probe during an interview.
        5.  **Work Experience Validation:** Analyze the 'Skills Demonstrated in Work Experience'. This list shows required skills the candidate has verifiably used in a professional context, indicating practical competence.
        ### Formatting Rules:
        - Use an objective, analytical, and professional tone. Use clean headings and write like you are providing analysis to a HR for candidate
        - DO NOT use a letter format or markdown asterisks (`**`).
        """

    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
    report_text = response.choices[0].message.content.replace('**', '')
    return markdown(report_text)

# =========================
# RUN THE APPLICATION
# =========================
if __name__ == "__main__":
    app.run(debug=True)




