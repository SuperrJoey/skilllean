
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="SkillLens AI",
    page_icon="🎯",
    layout="wide",
)

# Skills that SkillLens can identify.
SKILLS = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "llm", "generative ai",
    "agentic ai", "pandas", "numpy", "matplotlib", "scikit-learn",
    "tensorflow", "pytorch", "data analysis", "data visualization",
    "statistics", "excel", "power bi", "tableau", "git", "github",
    "flask", "fastapi", "streamlit", "react", "node.js",
    "communication", "problem solving", "teamwork",
    "financial analysis", "risk management", "finance",
]


def find_skills(text):
    """Find known skills using case-insensitive text matching."""
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(found)


def calculate_similarity(resume, job_description):
    """Calculate TF-IDF cosine similarity between two texts."""
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job_description])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)


st.title("🎯 SkillLens AI")
st.subheader("Resume and Job Description Analyzer")

st.write(
    "Compare a resume with a job description using NLP, "
    "identify matching skills, and discover potential skill gaps."
)

st.info(
    "The similarity score measures text overlap. It is not a "
    "probability of getting hired or a measure of overall suitability."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Your resume")
    resume = st.text_area(
        "Paste resume text",
        height=260,
        placeholder=(
            "Example: B.Tech CSE student with Python, Pandas, "
            "NumPy and machine learning experience..."
        ),
    )

with col2:
    st.markdown("### Target job")
    job_description = st.text_area(
        "Paste job description",
        height=260,
        placeholder=(
            "Example: Looking for candidates with Python, "
            "machine learning, data analysis and NLP skills..."
        ),
    )

if st.button("Analyze Resume", type="primary", use_container_width=True):

    if not resume.strip() or not job_description.strip():
        st.error("Please enter both your resume and the job description.")

    elif len(resume.split()) < 5 or len(job_description.split()) < 5:
        st.error("Please provide more text in both fields.")

    else:
        resume_skills = find_skills(resume)
        job_skills = find_skills(job_description)

        matching_skills = sorted(
            set(resume_skills) & set(job_skills)
        )

        missing_skills = sorted(
            set(job_skills) - set(resume_skills)
        )

        similarity = calculate_similarity(
            resume,
            job_description,
        )

        skill_coverage = (
            len(matching_skills) / len(job_skills) * 100
            if job_skills else 0
        )

        st.divider()
        st.markdown("## Analysis Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Text Similarity",
            f"{similarity:.1f}%",
        )

        metric2.metric(
            "Matching Skills",
            len(matching_skills),
        )

        metric3.metric(
            "Detected Skill Coverage",
            f"{skill_coverage:.1f}%",
        )

        st.progress(skill_coverage / 100)

        left, right = st.columns(2)

        with left:
            st.markdown("### ✅ Matching skills")

            if matching_skills:
                for skill in matching_skills:
                    st.write(f"• {skill}")
            else:
                st.write("No matching skills detected.")

        with right:
            st.markdown("### 📌 Potential skill gaps")

            if missing_skills:
                for skill in missing_skills:
                    st.write(f"• {skill}")
            else:
                st.write("No gaps detected in the supported skill list.")

        st.markdown("### Skill comparison table")

        all_skills = sorted(set(resume_skills) | set(job_skills))

        comparison = pd.DataFrame({
            "Skill": all_skills,
            "In Resume": [
                "Yes" if skill in resume_skills else "No"
                for skill in all_skills
            ],
            "In Job Description": [
                "Yes" if skill in job_skills else "No"
                for skill in all_skills
            ],
        })

        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.caption(
            "Skill extraction uses a predefined vocabulary and exact "
            "phrase matching. It may miss synonyms, context, and skills "
            "expressed in different ways. Review the results manually."
        )
