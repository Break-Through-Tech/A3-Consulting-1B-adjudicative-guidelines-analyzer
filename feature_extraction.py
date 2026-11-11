import pandas as pd
from pathlib import Path

# Resolve local generated splits dynamically across any folder layout
script_dir = Path(__file__).resolve().parent

# Team's processed splits folder
train_data_path = script_dir / "data" / "processed" / "splits" / "train.parquet"

# Root data folder layout
if not train_data_path.exists():
    train_data_path = script_dir / "data" / "train.parquet"

#Notebooks folder layout
if not train_data_path.exists():
    train_data_path = script_dir / "notebooks" / "data" / "train.parquet"

# Final safety check
if not train_data_path.exists():
    raise FileNotFoundError(f"Could not locate train.parquet. Inspected path: {train_data_path}")

print(f"Loading repository data from: {train_data_path}")
df = pd.read_parquet(train_data_path)


#KEYWORD HEURISTICS (Descriptive Box)
# This loops through the 'full_text' column to spot red flags automatically
def apply_keyword_heuristics(text):
    text_lower = str(text).lower()
    tags = []
    
    # Guideline A: Allegiance to the United States
    if any(word in text_lower for word in ["allegiance", "sabotage", "espionage", "treason", "terrorism", "overthrow", "subversive"]):
        tags.append("Guideline A: Allegiance")
        
    # Guideline B: Foreign Influence
    if any(word in text_lower for word in ["foreign", "citizenship", "passport", "overseas", "dual citizen", "foreign national"]):
        tags.append("Guideline B: Foreign Influence")
        
    # Guideline C: Foreign Preference
    if any(word in text_lower for word in ["foreign preference", "foreign military", "foreign passport", "allegiance to foreign"]):
        tags.append("Guideline C: Foreign Preference")
        
    # Guideline D: Sexual Behavior
    if any(word in text_lower for word in ["sexual", "coercion", "blackmail", "sexual conduct", "prostitution"]):
        tags.append("Guideline D: Sexual Behavior")
        
    # Guideline E: Personal Conduct
    if any(word in text_lower for word in ["falsified", "omission", "concealed", "untruthful", "lied", "refused to cooperate"]):
        tags.append("Guideline E: Personal Conduct")
        
    # Guideline F: Financial Considerations
    if any(word in text_lower for word in ["debt", "bankruptcy", "overdue", "financial", "credit", "delinquent", "foreclosure", "tax"]):
        tags.append("Guideline F: Financial")
        
    # Guideline G: Alcohol Consumption
    if any(word in text_lower for word in ["alcohol", "dui", "dwi", "intoxicated", "drunk", "alcoholism", "rehabilitation"]):
        tags.append("Guideline G: Alcohol")
        
    # Guideline H: Drug Involvement and Substance Misuse
    if any(word in text_lower for word in ["drug", "marijuana", "cocaine", "controlled substance", "substance misuse", "heroin", "amphetamine"]):
        tags.append("Guideline H: Drug Involvement")
        
    # Guideline I: Psychological Conditions
    if any(word in text_lower for word in ["psychological", "mental health", "emotional", "judgment", "disorder", "psychiatrist"]):
        tags.append("Guideline I: Psychological")
        
    # Guideline J: Criminal Conduct
    if any(word in text_lower for word in ["arrest", "criminal", "misdemeanor", "police", "illegal", "felony", "conviction", "jail"]):
        tags.append("Guideline J: Criminal")
        
    # Guideline K: Handling Protected Information
    if any(word in text_lower for word in ["classified", "protected information", "security leak", "unauthorized disclosure", "mishandling"]):
        tags.append("Guideline K: Protected Info")
        
    # Guideline L: Outside Activities
    if any(word in text_lower for word in ["outside employment", "conflict of interest", "unapproved activity", "foreign service"]):
        tags.append("Guideline L: Outside Activities")
        
    # Guideline M: Use of Information Technology
    if any(word in text_lower for word in ["hacking", "unauthorized access", "it systems", "cyber", "network breach", "malware"]):
        tags.append("Guideline M: IT Use")
        
    return tags if tags else ["No specific guideline flagged"]


# AI PROMPT TEMPLATE GENERATOR (RAG Box)
# This serves as the master instruction card our team's RAG system will use (if need be)
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

