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

# =========================
# CONFIG & DETERMINISTIC SETUP
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Force deterministic results for consistent scores
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = Flask(__name__)
app.secret_key = "supersecret"  # Change in production
client = openai.OpenAI(api_key=api_key)

MODEL_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CODEBERT = "microsoft/codebert-base"
MODEL_GRAPHCODEBERT = "microsoft/graphcodebert-base"
ONTOLOGY_PATH = "skills_ontology.json"

# =========================
# HELPERS
# =========================

def clean_text(text: str) -> str:
    """Basic cleaning of OCR/PDF text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text).strip()
    return text.lower()

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF, using OCR as a fallback for image-based pages."""
    text = ""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    doc.close()
    return clean_text(text)

def load_ontology(path=ONTOLOGY_PATH):
    """Loads the skill ontology from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_skill_maps(ontology):
    """
    NEW: Creates a flat list of all skills/aliases and a map to find the canonical skill.
    This allows detecting skills like 'machine learning' even if they are values in the ontology.
    """
    all_skills_and_aliases = set()
    alias_to_canonical_map = {}
    for canonical_skill, aliases in ontology.items():
        key_lower = canonical_skill.lower()
        all_skills_and_aliases.add(key_lower)
        alias_to_canonical_map[key_lower] = key_lower
        for alias in aliases:
            alias_lower = alias.lower()
            all_skills_and_aliases.add(alias_lower)
            alias_to_canonical_map[alias_lower] = key_lower
    return list(all_skills_and_aliases), alias_to_canonical_map

def extract_skills(text, all_skills_list, alias_map):
    """
    REWRITTEN: Extracts canonical skills using whole-word matching against the full skill list.
    """
    found_canonical_skills = set()
    for skill in all_skills_list:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_canonical_skills.add(alias_map[skill])
    return list(found_canonical_skills)

def embed_text(texts, model_name):
    """
    MODIFIED: Now processes one model at a time to avoid truncation issues.
    """
    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb.reshape(1, -1) # Ensure 2D array for cosine similarity

# Replace your old get_similarity function with this one
def get_similarity(a, b):
    """Calculates cosine similarity and normalizes it to a 0-1 range."""
    score = cosine_similarity(a, b)[0][0]
    # Normalize the score from [-1, 1] to [0, 1]
    normalized_score = (score + 1) / 2
    return float(normalized_score)

def generate_report(jd_skills, resume_skills, missing_skills, missing_project_skills):
    """
    Generates a textual analysis report using OpenAI GPT.
    MODIFIED: Added temperature=0 for consistent output.
    """
    prompt = f"""
    You are an expert AI resume analyzer providing feedback to a candidate.

    ### Analysis Input:
    - **Required Skills (from Job Description):** {", ".join(jd_skills) if jd_skills else "None"}
    - **Candidate's Skills (from Resume):** {", ".join(resume_skills) if resume_skills else "None"}
    - **Candidate's Missing Core Skills:** {", ".join(missing_skills) if missing_skills else "None"}
    - **Skills Missing from Projects:** {", ".join(missing_project_skills) if missing_project_skills else "None"}

    ### Your Task:
    1.  **Summarize Strengths:** Briefly summarize the candidate's skills that align with the job description.
    2.  **Analyze Gaps:** Clearly explain the missing skills and why they are important for the role.
    3.  **Suggest Projects:** Provide 2-3 specific project ideas that would help the candidate learn the missing skills and strengthen their resume. Make the suggestions actionable and relevant.
    4.  **Maintain Tone:** Be professional, constructive, and motivating.

    Format your response using clean Markdown with bold headings and bullet points.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Makes the output deterministic and consistent
    )
    return response.choices[0].message.content

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def dashboard():
    report, scores, details = None, None, None

    if request.method == "POST":
        jd_text = clean_text(request.form.get("jd_text", ""))
        resume_file = request.files.get("resume")
        if not resume_file:
            return "No resume file provided.", 400

        resume_text = extract_text_from_pdf(resume_file)
        
        # --- 1. Skill Extraction (New Logic) ---
        ontology = load_ontology()
        all_skills_list, alias_map = create_skill_maps(ontology)
        
        jd_skills = extract_skills(jd_text, all_skills_list, alias_map)
        resume_skills = extract_skills(resume_text, all_skills_list, alias_map)

        # Extract text from specific resume sections for more targeted scoring
        projects_text = re.search(r"(?i)projects(.*?)(skills|education|experience|$)", resume_text, re.DOTALL)
        projects_text = projects_text.group(1) if projects_text else ""
        
        experience_text = re.search(r"(?i)experience(.*?)(skills|education|projects|$)", resume_text, re.DOTALL)
        experience_text = experience_text.group(1) if experience_text else ""
        
        qualifications_text = re.search(r"(?i)education|qualifications(.*?)(skills|experience|projects|$)", resume_text, re.DOTALL)
        qualifications_text = qualifications_text.group(1) if qualifications_text else ""

        # --- 2. Scoring by Averaging Scores (New Logic) ---
        models = [MODEL_SBERT, MODEL_CODEBERT, MODEL_GRAPHCODEBERT]
        jd_skills_text = " ".join(jd_skills)

        # Calculate scores for each category by averaging model outputs
        def calculate_average_score(section_text, target_text):
            if not section_text or not target_text:
                return 0.0
            model_scores = []
            for model in models:
                emb_section = embed_text([section_text], model)
                emb_target = embed_text([target_text], model)
                model_scores.append(get_similarity(emb_section, emb_target))
            return np.mean(model_scores) if model_scores else 0.0

        skills_score = calculate_average_score(" ".join(resume_skills), jd_skills_text)
        projects_score = calculate_average_score(projects_text, jd_skills_text)
        experience_score = calculate_average_score(experience_text, jd_skills_text)
        qualifications_score = calculate_average_score(qualifications_text, "bachelor master phd degree education")

        overall = np.mean([skills_score, projects_score, experience_score, qualifications_score])

        # --- 3. Details for Report ---
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))
        extra_skills = sorted(list(set(resume_skills) - set(jd_skills)))
        
        project_skills_found = extract_skills(projects_text, all_skills_list, alias_map)
        missing_project_skills = sorted(list(set(jd_skills) - set(project_skills_found)))
        projects_text = re.search(r"(?i)projects(.*?)(skills|education|experience|$)", resume_text, re.DOTALL)
        projects_text = projects_text.group(1) if projects_text else ""
        print("---- EXTRACTED PROJECTS TEXT ----")
        print(projects_text)
        print("---------------------------------")
        scores = {
            "skills": skills_score ,
            "projects": projects_score ,
            "experience": experience_score ,
            "qualifications": qualifications_score,
            "overall": overall
        }
        details = {
            "jd_skills": jd_skills,
            "resume_skills": resume_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "missing_project_skills": missing_project_skills
        }
        
        report = generate_report(jd_skills, resume_skills, missing_skills, missing_project_skills)

    return render_template("dashboard.html", scores=scores, details=details, report=report)

if __name__ == "__main__":
    app.run(debug=True)










