---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, which will allow you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---
## 📋 BTT Internal Evaluation Notes

| Check   | Status | Notes                                                                                          |
|---------|--------|------------------------------------------------------------------------------------------------|
| Python Compatibility | 🟢  | The project utilizes Python-compatible tools and libraries throughout the tech stack, making it suitable for the students' current skill level. |
| Data Readiness | 🟡 | The datasets are substantially populated, though potential nuances in the documents may require cleaning; students should expect to invest time in data preparation. |
| Resource Check | 🟢  | No proprietary software requirements; all tools and environments are freely accessible to students via cloud resources. |

**Student Fit Score:** 7/10  
**Technical Depth Score:** 8/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
Strengths include a clearly defined use case within a relevant domain for AI applications, providing students with a strong real-world context. However, students should be prepared for extensive work in NLP and the unique challenges associated with the security-clearance context. For improvements: 1) Simplify document processing requirements to ensure students can focus on modeling and evaluation. 2) Consider adding scaffolding for LLM interactions to aid students who may not be comfortable with complex model implementations. Overall, encourage the team to engage in close iterative collaboration with faculty for optimal success.

---

# Explainable AI for Security Clearance Decision

**Company / Org:** A3 Consulting LLC  
**Challenge Advisor:** Teneika Askew, teneika.askew@a3consultingllc.com  
**AI Studio Coach:** Shaun Figueiro, shaun.figueiro@breakthroughtech.org                 
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About A3 Consulting LLC

A3 Consulting LLC specializes in providing innovative solutions tailored to the security clearance domain, utilizing data-driven methodologies to enhance decision-making processes within governmental and defense sectors.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use ~36,700 publicly available DOHA (Defense Office of Hearings and Appeals) security clearance decision documents and a combination of NLP, large language models, and retrieval-augmented generation (RAG) to build a system that classifies clearance case narratives against the 13 SEAD-4 adjudicative guidelines (A through M), detects disqualifying and mitigating conditions with specific paragraph citations, predicts the likely outcome (granted/denied), and retrieves similar precedent cases. This will help our company address the slow, inconsistent, and labor-intensive nature of security clearance adjudication for federal defense and intelligence customers by providing fast, explainable, precedent-grounded decision support.

### Success Criteria
- Quantitative: outcome-prediction performance (accuracy, F1, AUC) against held-out cases, with attention to class balance since most cases skew toward one outcome; multi-label guideline classification (per-label F1, micro/macro averages); and retrieval quality (precision@k on whether retrieved precedents share guidelines/outcome with the query case).    
- Qualitative: a successful December outcome is a working, explainable pipeline where the model's cited disqualifiers and mitigators are traceable to actual SEAD-4 paragraphs, the precedent retrieval surfaces genuinely analogous cases, and the team can articulate where the model is and is not trustworthy.

### Stretch Goals
Calibration analysis on the confidence scores. Fairness/consistency auditing (do similar fact patterns get similar predictions across years or guidelines?). Fine-tuning a small open model on the corpus and comparing it to the prompted-LLM approach. Severity-scoring regression (A through D scale). Temporal drift analysis on how adjudication patterns shift across 2016 to 2026. An "explanation quality" rubric scored by a held-out LLM judge.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month     | Milestone             | Key Activities                                                      |
|-----------|-----------------------|---------------------------------------------------------------------|
| **September** | [Title] | Onboard team onto the provided DOHA dataset (parsed Parquet, no scraping needed). Exploratory data analysis on case outcomes, guideline frequency, and text characteristics. Establish a baseline classification model (TF-IDF plus logistic regression / lightweight transformer) predicting granted vs. denied, and a labeled subset for evaluation.          |
| **October** | [Title] | Build the guideline-classification component (multi-label A through M) and the LLM-based disqualifier/mitigator extraction using the free-tier Gemini API. Stand up the RAG precedent-retrieval pipeline against the provided vector index. Compare LLM-based vs. traditional ML approaches on the same eval set.         |
| **November** | [Title] | Integrate components into an end-to-end analyzer that outputs a structured assessment (recommendation, confidence, guideline breakdown, cited precedents). Build a Streamlit demo UI. Run final evaluation, error analysis, and document limitations. Prepare presentation and GitHub deliverable.            |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** DOHA Security Clearance Decision Documents  
**Format:** CSV, JSON, Parquet, PDF  
**Size:** 5gb to 10gb  
**Location:** https://github.com/TeneikaAskew/doha

### Key Details
- ~36,700 publicly available DOHA (Defense Office of Hearings and Appeals) documents containing numerical, categorical, and text data, stored in CSV, JSON, Parquet, and PDF formats.
- Some documents may require preprocessing to handle inconsistencies in formatting or missing information.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- Accuracy, Precision/Recall, F1 score, AUC

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Link to an article or blog post about the problem domain]
- [Link to an industry report or case study]

**Technical Tutorials:**
- [Link to a free tutorial on the ML technique(s) involved]
- [Link to documentation for a key library or tool]

**Code Examples:**
- [Link to a relevant GitHub repo]
- [Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session C).
