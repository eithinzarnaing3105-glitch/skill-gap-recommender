import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core logic (same as the notebook)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("🎯 Skill-Gap Recommender")
st.write(
    "Find out which skills to learn next, based on how AI and traditional IT "
    "skills co-occur across real IT job postings."
)

with st.form("recommender_form"):
    skills_input = st.text_input(
        "Your current skills (comma-separated)",
        placeholder="e.g. SQL, Excel",
    )
    role_input = st.text_input(
        "Your target IT role",
        placeholder="e.g. Data Analyst",
    )
    submitted = st.form_submit_button("Get my recommendation")

if submitted:
    if not skills_input.strip() or not role_input.strip():
        st.warning("Please fill in both your current skills and your target role.")
    else:
        current_skills = [s.strip() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing skill patterns and generating your recommendation..."):
            recommendation, relevant_rules = get_recommendation(
                current_skills=current_skills,
                target_role=role_input.strip(),
                rules_df=rules_df,
            )

        st.subheader("Your recommendation")
        st.write(recommendation)

        if not relevant_rules.empty:
            with st.expander("See the skill patterns this was based on"):
                st.dataframe(
                    relevant_rules[["Premises", "Conclusion", "Confidence", "Lift"]],
                    hide_index=True,
                )