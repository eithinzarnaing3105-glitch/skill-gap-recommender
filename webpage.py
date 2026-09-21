import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="Skill-Gap Recommender", page_icon="🎯")


@st.cache_data
def load_rules():
    rules_df = pd.read_csv("Association_rules.csv")
    rules_df.columns = [c.strip() for c in rules_df.columns]
    return rules_df


rules_df = load_rules()



def split_skills(cell):
    if pd.isna(cell):
        return []
    return [s.strip().lower() for s in str(cell).split(",")]


def get_relevant_rules(current_skills, rules_df, top_n=8):
    current_skills_lower = set(s.strip().lower() for s in current_skills)

    def overlaps(premises_cell):
        premise_skills = set(split_skills(premises_cell))
        return len(premise_skills & current_skills_lower) > 0

    matched = rules_df[rules_df["Premises"].apply(overlaps)].copy()
    matched = matched.sort_values(by=["Confidence", "Lift"], ascending=False)
    return matched.head(top_n)


def build_prompt(current_skills, target_role, relevant_rules):
    rules_text = "\n".join(
        f"- If someone has [{row['Premises']}], job postings often also require [{row['Conclusion']}] "
        f"(confidence={row['Confidence']:.2f}, lift={row['Lift']:.2f})"
        for _, row in relevant_rules.iterrows()
    )

    prompt = f"""You are a career advisor for IT students.

A student currently has these skills: {', '.join(current_skills)}.
Their target job role is: {target_role}.

Here are real skill co-occurrence patterns found in IT job postings, from association rule mining:
{rules_text}

Based on this evidence, recommend the 3 to 5 most important skills the student should learn next.
For each skill, give a one-sentence, plain-language reason tied to the patterns above.
Keep the whole answer short, friendly, and easy for a student to read — no jargon, no long paragraphs.
"""
    return prompt


def get_recommendation(current_skills, target_role, rules_df, top_n=8, max_retries=3):
    relevant = get_relevant_rules(current_skills, rules_df, top_n=top_n)

    if relevant.empty:
        return "No matching skill patterns were found for these inputs. Try different or more general current skills.", relevant

    prompt = build_prompt(current_skills, target_role, relevant)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return response.text, relevant
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return f"Failed after {max_retries} attempts: {e}", relevant

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=Sora:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
 
    :root {
        --bg: #e8dfcf;
        --panel: #FBFAF7;
        --panel-border: #DEDACF;
        --ink: #23241F;
        --cobalt: #2554C7;
        --coral: #E8703C;
        --muted: #6E6B60;
    }
 
    .stApp {
        background-color: var(--bg);
        background-image:
            repeating-radial-gradient(circle at 12% 18%, rgba(37, 84, 199, 0.06) 0px, rgba(37, 84, 199, 0.06) 1px, transparent 1px, transparent 34px),
            repeating-radial-gradient(circle at 88% 78%, rgba(232, 112, 60, 0.07) 0px, rgba(232, 112, 60, 0.07) 1px, transparent 1px, transparent 30px),
            repeating-radial-gradient(circle at 85% 8%, rgba(37, 84, 199, 0.04) 0px, rgba(37, 84, 199, 0.04) 1px, transparent 1px, transparent 46px);
        background-attachment: fixed;
        background-size: 200% 200%, 220% 220%, 180% 180%;
        animation: driftLight 34s ease-in-out infinite;
        color: var(--ink);
        font-family: 'Sora', sans-serif;
    }
 
    @keyframes driftLight {
        0%   { background-position: 0% 0%, 100% 100%, 100% 0%; }
        50%  { background-position: 4% 6%, 94% 92%, 96% 4%; }
        100% { background-position: 0% 0%, 100% 100%, 100% 0%; }
    }
 
    #MainMenu, footer, header { visibility: hidden; }
 
    .block-container {
        max-width: 640px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }
 
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
 
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 1.15rem;
        letter-spacing: 0.01em;
        color: var(--cobalt);
        margin-bottom: 1.1rem;
        animation: fadeUp 0.6s ease-out both;
    }
 
    .hero-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 700;
        font-size: 2.7rem;
        line-height: 1.14;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin-bottom: 1rem;
        animation: fadeUp 0.6s ease-out 0.08s both;
    }
 
    .hero-title .accent {
        background: linear-gradient(120deg, var(--cobalt), var(--coral));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
 
    .hero-sub {
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--ink);
        max-width: 44ch;
        margin-bottom: 2.4rem;
        animation: fadeUp 0.6s ease-out 0.16s both;
    }
 
    .stTextInput {
        animation: fadeUp 0.6s ease-out 0.22s both;
    }
 
    .stTextInput label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--ink);
    }
 
    .stTextInput input {
        background-color: var(--panel) !important;
        color: var(--ink) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 10px !important;
        padding: 0.7rem 0.9rem !important;
        font-family: 'Sora', sans-serif !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
 
    .stTextInput input:focus {
        border-color: transparent !important;
        box-shadow: 0 0 0 2px var(--cobalt), 0 0 14px rgba(37, 84, 199, 0.18) !important;
    }
 
    .stTextInput input::placeholder {
        color: #A39D8E !important;
    }
 
    .stFormSubmitButton button {
        background: linear-gradient(120deg, var(--cobalt), #e08d63) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.6rem !important;
        margin-top: 0.6rem;
        box-shadow: 0 3px 14px rgba(37, 84, 199, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeUp 0.6s ease-out 0.28s both;
    }
 
    .stFormSubmitButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(37, 84, 199, 0.3);
    }
 
    .results-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--cobalt);
        margin: 2rem 0 1rem 0;
        animation: fadeUp 0.5s ease-out both;
    }
 
    .skill-card {
        display: flex;
        gap: 1rem;
        align-items: baseline;
        padding: 1.1rem 0;
        border-bottom: 1px solid var(--panel-border);
        opacity: 0;
        animation: fadeUp 0.45s ease-out forwards;
    }
 
    .skill-card:first-of-type {
        border-top: 1px solid var(--panel-border);
    }
 
    .skill-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: var(--coral);
        min-width: 1.6rem;
        flex-shrink: 0;
    }
 
    .skill-body {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
 
    .skill-name {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 600;
        font-size: 1.18rem;
        color: var(--ink);
        line-height: 1.3;
    }
 
    .skill-reason {
        font-family: 'Sora', sans-serif;
        font-size: 0.96rem;
        line-height: 1.55;
        color: var(--muted);
    }
 
    .fallback-text {
        font-family: 'Sora', sans-serif;
        font-size: 1.02rem;
        line-height: 1.7;
        color: var(--ink);
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-left: 3px solid var(--coral);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1.6rem 0;
        white-space: pre-wrap;
    }
 
    .evidence-item {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        padding: 0.7rem 0;
        border-bottom: 1px solid var(--panel-border);
        font-size: 0.92rem;
        color: var(--ink);
        opacity: 0;
        animation: fadeUp 0.4s ease-out forwards;
    }
 
    .evidence-pair {
        max-width: 68%;
    }
 
    .evidence-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--coral);
        white-space: nowrap;
    }
 
    .stExpander {
        border: 1px solid var(--panel-border) !important;
        border-radius: 12px !important;
        background-color: var(--panel) !important;
    }
 
    .stExpander summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        color: var(--muted) !important;
    }
 
    .stAlert {
        background-color: var(--panel) !important;
        color: var(--ink) !important;
        border-radius: 10px !important;
        border: 1px solid var(--panel-border) !important;
    }
 
    .stSpinner > div {
        border-top-color: var(--cobalt) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">Skill-Gap Recommender 🤖 </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Find your next skill <span class="accent">before the job posting does</span>.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Enter what you know and where you\'re headed. '
    "We'll map the skill patterns from real IT job postings and show what's worth learning next.</div>",
    unsafe_allow_html=True,
)
 
with st.form("recommender_form"):
    skills_input = st.text_input(
        "Skills you already have",
        placeholder="SQL, Excel",
    )
    role_input = st.text_input(
        "Role you're aiming for",
        placeholder="Data Analyst",
    )
    submitted = st.form_submit_button("Reveal my next skills")
 
if submitted:
    if not skills_input.strip() or not role_input.strip():
        st.warning("Add both your current skills and your target role to continue.")
    else:
        current_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
 
        with st.spinner("Mapping the skill network…"):
            recommendation, relevant_rules = get_recommendation(
                current_skills=current_skills,
                target_role=role_input.strip(),
                rules_df=rules_df,
            )
 
        st.markdown(
            f'<div class="results-label">toward {role_input.strip().lower()}</div>',
            unsafe_allow_html=True,
        )
 
        if isinstance(recommendation, list):
            for i, item in enumerate(recommendation):
                delay = 0.08 * i
                st.markdown(
                    f"""<div class="skill-card" style="animation-delay: {delay}s;">
                        <span class="skill-index">{i + 1:02d}</span>
                        <div class="skill-body">
                            <span class="skill-name">{item.get('skill', '')}</span>
                            <span class="skill-reason">{item.get('reason', '')}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f'<div class="fallback-text">{recommendation}</div>', unsafe_allow_html=True)
 
        if not relevant_rules.empty:
            with st.expander("View the evidence this was based on"):
                for i, (_, row) in enumerate(relevant_rules.iterrows()):
                    delay = 0.06 * i
                    st.markdown(
                        f"""<div class="evidence-item" style="animation-delay: {delay}s;">
                            <span class="evidence-pair">{row['Premises']} → {row['Conclusion']}</span>
                            <span class="evidence-metric">conf {row['Confidence']:.2f} · lift {row['Lift']:.2f}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )