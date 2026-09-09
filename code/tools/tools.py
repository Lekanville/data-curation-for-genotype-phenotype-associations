import os
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
UMLS_API_KEY = os.getenv("UMLS_API_KEY")

def fetch_hpo_names(hp_id):
    url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{hp_id}?format=json"
    try:
        response = requests.get(url, headers={'accept': 'application/json'}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return hp_id, data.get("name", "")
    except Exception as e:
        # Return empty string or None on failure
        return hp_id, ""

def batch_fetch_hpo_names(hpo_ids, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fetch_hpo_names, hpo_ids), total=len(hpo_ids), desc="Fetching HPO Names"))
    return dict(results)




def fetch_related_phenotypes(mondo_id):
    encoded_id = mondo_id.replace(":", "%3A")
    # url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{encoded_id}?format=json"
    # To get the actual phenotypes associated with the disease
    url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{encoded_id}/biolink%3ADiseaseToPhenotypicFeatureAssociation"
    
    try:
        response = requests.get(url, headers={'accept': 'application/json'})
        response.raise_for_status()
        data = response.json()
        
        relationships = []
        for item in data.get('items', []):
            if item.get("object", "").startswith("HP:"):
                relationship = {
                    # "disease_id": item.get("subject"),
                    # "disease_name": item.get("subject_label"),
                    # "relationship": item.get("predicate"),
                    'HPO_ID': item.get("object"),
                    'Phenotype_Name': item.get("object_label")
                }
                relationships.append(relationship)
            
        print(f"Successfully fetched {len(relationships)} phenotype relationships for {mondo_id}")
        return relationships
    except Exception as e:
        print(f"❌ Error fetching {mondo_id}: {e}")
        return []



@lru_cache(maxsize=None)
def get_umls_cui_with_retry(source_code, source_vocab, input_type="sourceUi", max_retries=5):
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "apiKey": UMLS_API_KEY,
        "string": str(source_code),
        "sabs": source_vocab,
        "searchType": "exact"
    }
    # Only send inputType if specified
    if input_type:
        params["inputType"] = input_type

    backoff = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 200:
                results = response.json().get("result", {}).get("results", [])
                if results:
                    return results[0]["ui"]
                return np.nan
        except requests.RequestException:
            time.sleep(backoff)
            backoff *= 2
    return np.nan


def batch_get_cui_safe(codes, vocab, input_type="sourceUi", max_workers=5):
    def fetch(code):
        return str(code), get_umls_cui_with_retry(str(code), vocab, input_type=input_type)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fetch, codes), total=len(codes), desc=f"Mapping {vocab}"))
    
    return dict(results)




def get_hpo_id_from_cui(cui):
    """
    Fetches the HPO ID associated with a UMLS CUI.
    """
    if not cui or cui == "None" or pd.isna(cui):
        return None
        
    if not UMLS_API_KEY:
        raise ValueError("UMLS_API_KEY is not set in the environment.")

    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"
    params = {
        "apiKey": UMLS_API_KEY,
        "sabs": "HPO"  # Restrict search results to Human Phenotype Ontology
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("result", [])
            for atom in results:
                code = atom.get("code", "")
                if "HPO/" in code:
                    hpo_code = code.split("HPO/")[-1]
                    return f"HP:{hpo_code}" if not hpo_code.startswith("HP:") else hpo_code
                elif code.startswith("HP:"):
                    return code
    except Exception:
        pass
    return None

def batch_get_hpo_from_cuis(cuis, max_workers=5):
    """
    Multithreaded fetcher to map CUIs to HPO IDs while respecting UMLS rate limits.
    """
    def fetch(cui):
        return cui, get_hpo_id_from_cui(cui)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(fetch, cuis), 
            total=len(cuis), 
            desc="Mapping CUIs to HPO IDs"
        ))
    
    return dict(results)