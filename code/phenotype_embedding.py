import os
import re
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_umls_api_key() -> str:
    load_dotenv()
    umls_api_key = os.getenv("UMLS_API_KEY")
    if not umls_api_key:
        raise ValueError("UMLS_API_KEY is not set in the environment.")
    return umls_api_key


def create_session_with_retries():
  """Creates a requests session that automatically retries on rate limits (429) or server errors (500, 503)."""
  session = requests.Session()
  retries = Retry(
      total=5,
      backoff_factor=1.0,  # Wait 1s, 2s, 4s, etc. between retries
      status_forcelist=[429, 500, 502, 503, 504],
      raise_on_status=False,
  )
  adapter = HTTPAdapter(max_retries=retries)
  session.mount("https://", adapter)
  return session


def fetch_cui_data(cui):
    API_KEY = load_umls_api_key()
    version = "current"
    base_url = "https://uts-ws.nlm.nih.gov/rest"
    preferred_sources = ["HPO", "MSH", "NCI", "SNOMEDCT_US"]

    session = create_session_with_retries()
    params = {"apiKey": API_KEY}

    name = "Unknown"
    definition = "No definition available"

    try:
        # 1. Fetch concept details for preferred name
        concept_url = f"{base_url}/content/{version}/CUI/{cui}"
        resp = session.get(concept_url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            name = data.get("name", "Unknown")

        # 2. Fetch definitions
        def_url = f"{base_url}/content/{version}/CUI/{cui}/definitions"
        def_resp = session.get(def_url, params=params, timeout=15)
        if def_resp.status_code == 200:
            defs = def_resp.json().get("result", [])
        else:
            defs = []

        found = False
        for source in preferred_sources:
            for d in defs:
                if d.get("rootSource") == source:
                    definition = d.get("value", "No definition available")
                    found = True
                    break
            if found:
                break

        if not found and defs:
            for d in defs:
                src = d.get("rootSource", "")
                if not any(
                    src.endswith(lang)
                    for lang in ["CZE", "POR", "SPA", "FRE", "GER", "ITA"]
                ):
                    definition = d.get("value", "No definition available")
                    break

    except Exception as e:
        print(f"Error fetching {cui}: {e}")

    return {"cui": cui, "preferred_name": name, "definition": definition}

def clean_and_combine_outcome_and_defn(row):
    outcome = str(row["outcome"]).strip()
    definition = str(row["definition"]).strip()

    # 1. Clean out [PMID: ...], ORCIDs, and trailing/empty brackets like '[]'
    definition = re.sub(r"\s*\[\s*(?:PMID|orcid|http)[^\]]*\]", "", definition, flags=re.IGNORECASE)
    definition = re.sub(r"\s*\[\s*\]", "", definition)  # Removes empty [] brackets

    # 2. Handle missing or duplicate definitions
    if (
        not definition
        or definition.lower() == "no definition available"
        or definition.lower() == outcome.lower()
        ):
        return outcome
    else:
        return f"{outcome}: {definition.strip()}"

def build_phenotype_embeddings(clinical_outcomes_with_ic: str, output_dir: str) -> str:
    df = pd.read_csv(clinical_outcomes_with_ic)
    cuis = list(dict.fromkeys(df[df["cui"].str.startswith("C")]["cui"]))

    results = []
    max_workers = 4


    print(
        f"Fetching data for {len(cuis)} CUIs with auto-retry and {max_workers}"
        " threads..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cui = {
                executor.submit(fetch_cui_data, cui): cui for cui in cuis
            }

    for future in tqdm(as_completed(future_to_cui), total=len(cuis)):
        results.append(future.result())

    df_definitions = pd.DataFrame(results)

    df_full_defs = pd.merge(df, df_definitions, on="cui", how="outer")
    df_full_defs["definition"] = df_full_defs["definition"].replace("No definition available", np.nan)
    df_full_defs["definition"] = (df_full_defs["definition"].fillna(df_full_defs["outcome"]))
    df_full_defs = df_full_defs.drop(["preferred_name", "has_side_effect_stats"],axis=1, errors="ignore")

    df_full_defs["outcome_and_def"] = df_full_defs.apply(clean_and_combine_outcome_and_defn, axis=1)
    docs = df_full_defs["outcome_and_def"].tolist()

    embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings", device="cpu")
    embeddings = embedder.encode(docs, show_progress_bar=True)
    df_full_defs["embedding"] = list(embeddings)

    output_path = os.path.join(output_dir, "clinical_outcomes_with_embedding.parquet")
    df_full_defs.to_parquet(output_path, index=False)
    

    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Get phenotype description embedding")
    parser.add_argument("--clinical-outcomes-with-ic",required=True, help="Path to the clinical outcomes with IC CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the embedding output")
    args = parser.parse_args()

    output_path = build_phenotype_embeddings(args.clinical_outcomes_with_ic, args.output_dir)
    print("Clinical outcomes with embedding successfully saved")

if __name__ == "__main__":
    main()
