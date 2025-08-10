from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import re
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai

from LLMS.Edelweiss import conversational_rag_chain as edelweiss_rag_chain
from LLMS.HDFC import conversational_rag_chain as hdfc_rag_chain
from LLMS.LIC import conversational_rag_chain as lic_rag_chain
from LLMS.KOTAK import conversational_rag_chain as kotak_rag_chain
from LLMS.StarHealth import conversational_rag_chain as starhealth_rag_chain
from LLMS.Bajaj import conversational_rag_chain as bajaj_rag_chain

from LLMS.PolicyNames.bajaj_policy_names import bajaj_policy
from LLMS.PolicyNames.edelweiss_policy_names import edelweiss_policy
from LLMS.PolicyNames.hdfc_policy_names import hdfc_policy
from LLMS.PolicyNames.kotak_policy_names import kotak_policy
from LLMS.PolicyNames.lic_policy_names import lic_policy
from LLMS.PolicyNames.starhealth_policy_names import starhealth_policy

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

class PrettyJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        import json
        return json.dumps(content, indent=4, ensure_ascii=False).encode("utf-8")

app = FastAPI(default_response_class=PrettyJSONResponse)

# Models for input/output
class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "english"

class QueryResponse(BaseModel):
    answer: str

# Mapping chains and policies
company_rag_chains = {
    "edelweiss": edelweiss_rag_chain,
    "hdfc": hdfc_rag_chain,
    "lic": lic_rag_chain,
    "kotak": kotak_rag_chain,
    "starhealth": starhealth_rag_chain,
    "bajaj": bajaj_rag_chain,
}

company_policy_names = {
    "edelweiss": edelweiss_policy,
    "hdfc": hdfc_policy,
    "lic": lic_policy,
    "kotak": kotak_policy,
    "starhealth": starhealth_policy,
    "bajaj": bajaj_policy,
}

# Policy strings for prompt
policy_strings = {
    "edelweiss": "\n".join(f"- {p}" for p in edelweiss_policy),
    "hdfc": "\n".join(f"- {p}" for p in hdfc_policy),
    "lic": "\n".join(f"- {p}" for p in lic_policy),
    "kotak": "\n".join(f"- {p}" for p in kotak_policy),
    "starhealth": "\n".join(f"- {p}" for p in starhealth_policy),
    "bajaj": "\n".join(f"- {p}" for p in bajaj_policy),
}

def clean_strings(obj):
    if isinstance(obj, str):
        # Replace escaped newlines and double-escaped newlines
        return obj.replace("\\n", "\n").replace("\\\\n", "\n")
    elif isinstance(obj, dict):
        return {k: clean_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_strings(i) for i in obj]
    else:
        return obj


def normalize_question(raw_question: str) -> str:
    # Build the prompt for the generative model to normalize the question
    prompt = f"""
        You are an insurance query normalizer.
        Your task is to take a raw insurance query from a user and turn it into a clear, full English question with structured context.
        Raw input: "{raw_question}"
        These are the policies covered by HDFC RAG LLM: "{policy_strings['hdfc']}"
        These are the policies covered by Edelweiss RAG LLM: "{policy_strings['edelweiss']}"
        These are the policies covered by Bajaj RAG LLM: "{policy_strings['bajaj']}"
        These are the policies covered by KOTAK RAG LLM: "{policy_strings['kotak']}"
        These are the policies covered by LIC RAG LLM: "{policy_strings['lic']}"
        These are the policies covered by Star Health RAG LLM: "{policy_strings['starhealth']}"
        Return a **rephrased natural language question** that includes:
        - Age (if available)
        - Gender (if available)
        - Medical procedure
        - Location
        - Policy duration (if mentioned)
        - Context (e.g., whether asking about coverage, claim approval, or payout)
        ⚠️ If the user is asking “what is” or “tell me about” a specific policy name, instruct the RAG LLM to look up and return that exact policy's details.
        ⚠️ Encourage the RAG LLM to **search all available policy names and return relevant matching policy names** that may cover the user’s context (age, condition, etc.)..
        Make sure the result is a single, complete sentence or paragraph understandable by a retrieval/LLM system.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def query_llm(company: str, question: str, session_id="default_session") -> str:
    chain = company_rag_chains[company]
    policy_list = company_policy_names[company]
    policy_key = f"{company}_policy"

    response = chain.invoke(
        {
            "input": question,
            policy_key: "\n".join(f"- {p}" for p in policy_list),
        },
        config={"configurable": {"session_id": session_id}},
    )
    return response.get('answer', 'No answer returned.')


@app.post("/chat/{company_name}", response_model=QueryResponse)
async def chat_with_company(company_name: str, query: QueryRequest):
    company_name = company_name.lower()
    if company_name not in company_rag_chains:
        raise HTTPException(status_code=404, detail="Company not found")

    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Optional: check vague questions here and respond accordingly if needed
    # if is_vague(question):
    #     return QueryResponse(answer="It seems your question is a bit vague or lacking in context. Can you describe your question properly?")

    normalized_question = normalize_question(question)
    answer = query_llm(company_name, normalized_question)
    try:
        parsed_answer = json.loads(answer)
        cleaned_answer = clean_strings(parsed_answer)
        pretty_str = json.dumps(cleaned_answer, indent=4, ensure_ascii=False)
        return QueryResponse(answer=pretty_str)
    except Exception:
        # Fallback: return raw string in JSON field
        return QueryResponse(answer=answer)
    
