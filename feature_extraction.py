import pandas as pd
import os


# REPO PATH & FILE CHECK
train_data_path = os.path.join("data", "train.parquet")
if not os.path.exists(train_data_path):
    train_data_path = os.path.join("notebooks", "data", "train.parquet")

print(f"Loading repository data from: {train_data_path}")
df = pd.read_parquet(train_data_path)


#KEYWORD HEURISTICS (Descriptive Box)
# This loops through the 'full_text' column to spot red flags automatically
def apply_keyword_heuristics(text):
    text_lower = str(text).lower()
    tags = []
    
    # Financial Considerations (Guideline F)
    if any(word in text_lower for word in ["debt", "bankruptcy", "overdue", "financial", "credit"]):
        tags.append("Guideline F: Financial")
        
    # Criminal Conduct (Guideline J)
    if any(word in text_lower for word in ["arrest", "criminal", "misdemeanor", "police", "illegal"]):
        tags.append("Guideline J: Criminal")
        
    # Foreign Influence (Guideline B)
    if any(word in text_lower for word in ["foreign", "citizenship", "passport", "overseas"]):
        tags.append("Guideline B: Foreign")
        
    return tags if tags else ["No major flag detected"]


# AI PROMPT TEMPLATE GENERATOR (RAG Box)
# This serves as the master instruction card your team's RAG system will use
def get_llm_prompt_template(case_narrative):
    prompt_blueprint = f"""
    You are an expert security clearance adjudicator. 
    Read the following background check text narrative carefully:
    
    ---
    {case_narrative}
    ---
    
    Extract and list information cleanly using this exact structure:
    - RED FLAGS (Disqualifiers): List all negative security concerns or issues found.
    - EXCUSES (Mitigating Factors): List any context showing the issues are resolved or old.
    """
    return prompt_blueprint


#EXECUTE PIPELINE & GENERATE REPO FEATURES
print("Processing descriptive keyword mappings...")
df['keyword_tags'] = df['full_text'].apply(apply_keyword_heuristics)

# Quick Success Check Preview
print("\nSuccess! Part A (Keyword Features Loaded):")
print(df[['full_text', 'keyword_tags']].head(2))

print("\nSuccess! Part B (Master RAG Prompt Card Ready):")
# Take the first case narrative row in your repo files and test your template layout
sample_case = df['full_text'].iloc[0]
print(get_llm_prompt_template(sample_case[:300] + "... [Truncated for preview]"))