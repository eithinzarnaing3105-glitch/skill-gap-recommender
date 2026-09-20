# Skill-Gap Recommender for IT Students & Fresh Graduates

Recommending the next skill to learn, based on how AI and traditional IT skills co-occur across real IT job postings.

## Overview

As AI adoption spreads, many traditional IT roles (e.g., Data Analyst, System Administrator, QA Tester) are being reshaped to require AI/ML skills but it's unclear which roles are affected most, or which skill pairings are becoming common. Students and fresh graduates typically get generic career advice that misses this shift, so they don't know which skills are worth learning for their target IT role.

This project mines real IT job postings to find how AI/ML skills and traditional IT skills co-occur across different job roles, then uses an LLM to turn those patterns into a short, personalized "skills to learn next" recommendation for any IT career path, not just AI-specific roles.

This is a two-part project spanning two courses:

- **Data Mining course** — ETL and association rule mining (RapidMiner FP-Growth) on a real IT job postings dataset.
- **AI Capstone project** — an LLM layer (Gemini API) that turns the mined skill associations into plain-language, role-specific recommendations, delivered through a simple web app.

## How it works

```
IT Job Postings → Skill Extraction → Skill Association (FP-Growth) → LLM Recommender → Student Result Page
```

1. **Skill extraction** — Skills are parsed out of a large set of IT job postings, covering many roles (not only AI-focused ones).
2. **Association mining** — FP-Growth is used to find which traditional IT skills and AI/ML skills commonly appear together within each job role, and which non-AI roles show the most AI-related skill demand.
3. **LLM recommendation** — A student's current skills, target IT role, and the relevant mined associations are fed into the Gemini API, which generates a short, easy-to-read list of the next skills to learn and why.
4. **Result page** — A simple Streamlit web form lets a student enter their current skills and target role, and see their personalized recommendation along with the underlying skill patterns it was based on.

## Tech stack

| Layer | Tool |
|---|---|
| ETL & preprocessing | Python, pandas |
| Association rule mining | RapidMiner (FP-Growth + Create Association Rules) |
| Recommendation generation | Google Gemini API (`google-genai`) |
| Web interface | Streamlit |
| Config/secrets | python-dotenv |

## Dataset

IT job postings sourced from the [LinkedIn Jobs & Skills dataset on Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024), filtered down to IT-related roles and cleaned/categorized into job categories (Software Engineer/Developer, Data Analyst, QA, Network Engineer, DevOps/SRE, Security Engineer, etc.).

## Project structure

```
.
├── ETL.ipynb                  # Data cleaning & preprocessing (Data Mining course)
├── cleaned_jobs_dataset.csv   # Cleaned IT job postings dataset
├── Final project.rmp          # RapidMiner process: FP-Growth + Association Rules
├── Association_rules.csv      # Exported association rules (Premises, Conclusion, Support, Confidence, Lift)
├── Frequent_item_sets.csv     # Exported frequent item sets
├── skill_recommender.ipynb    # LLM recommendation logic & test cases (AI Capstone)
├── app.py                     # Streamlit web app (student-facing result page)
├── .env                       # Gemini API key (not committed)
└── README.md
```

## Setup

1. **Clone the repo and open it in your editor.**

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install google-genai pandas python-dotenv streamlit ipykernel
   ```

4. **Add your Gemini API key.** Get one from [Google AI Studio](https://aistudio.google.com), then create a `.env` file in the project root:
   ```
   GEMINI_API_KEY= "API_KEY"
   ```

## Usage

**Run the notebook** (for testing/exploring the recommendation logic):
Open `skill_recommender.ipynb` in VS Code or Jupyter, select the `venv` kernel, and run the cells in order.

**Run the web app:**
```bash
streamlit run app.py
```
This opens the app at `http://localhost:8501`. Enter your current skills and target IT role to get a personalized recommendation, along with the skill-association evidence it was based on.

## Research foundation

This project is based on and extends the approach in:

> Cheepmuangman, N., Viwathara, P., Pipattanasookmongkol, P., & Marukatat, R. (2024). *An Application of Text Mining and Association Rule Mining to Job and Skill Recommendations for IT Jobs.* Journal of Engineering and Digital Technology (JEDT), 12(1).

The original paper mines skills from IT job postings using Apriori and recommends new skills based on co-occurrence patterns. This project applies the same association-mining idea across different IT job roles to specifically compare AI-skill vs. traditional-skill co-occurrence, and adds an LLM layer to turn the mined associations into a personalized, plain-language recommendation instead of a fixed rule list.

## Limitations & future work

- Skill extraction relies on the categories present in the source dataset and may miss emerging or highly specific skill terms.
- Association rules are only as good as the `min_support` / `min_confidence` thresholds used in RapidMiner — some rare role/skill combinations may return no matching rules.
- Future improvements could include semantic skill matching (to catch skill synonyms), a larger/more recent job postings dataset, and deploying the Streamlit app for public access.

## Course context

Built as a two-course capstone project connecting Data Mining (association rule mining) and AI Capstone (LLM-based recommendation) coursework.
