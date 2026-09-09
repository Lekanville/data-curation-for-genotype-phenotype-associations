import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# --- Function to Query Monarch and Filter HP Terms ---
def fetch_hpo_closure(mondo_id):
    """Queries Monarch API and returns only HP: terms from the closure."""

    encoded_id = mondo_id.replace(":", "%3A")
    url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{encoded_id}?format=json"
    
    try:
        response = requests.get(url, headers={'accept': 'application/json'})
        response.raise_for_status()
        data = response.json()
        
        # Extract the closure list
        closure = data.get('has_phenotype_closure', [])
        
        # FILTERING LOGIC: Keep only HP IDs, ignore BFO, UPHENO, GO, etc.
        hp_terms = [term for term in closure if term.startswith("HP:")]
        
        print(f"Successfully fetched {len(hp_terms)} HP terms for {mondo_id}")
        return hp_terms

    except Exception as e:
        print(f"❌ Error fetching {mondo_id}: {e}")
        return []


def fetch_hpo_names(hp_id):
    """Queries Monarch API and returns only HP: terms from the closure."""
    encoded_id = hp_id
    url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{encoded_id}?format=json"
    
    try:
        response = requests.get(url, headers={'accept': 'application/json'})
        response.raise_for_status()
        data = response.json()
        
        # Extract the closure list
        hp_term_name = data.get("name", [])
        
        return hp_term_name
    
    except Exception as e:
        print(f"❌ Error fetching {hp_id}: {e}")
        return []

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

def map_to_anc_id(source_str):
    parts = source_str.split('_')
    if len(parts) > 1:
        anc_code = parts[-1] # Gets 'EUR', 'AFR', etc.
        return anc_code
    return source_str


def build_phenotype_features(
    phenotypes_with_prevalence: Path,
    new_phenotypes_from_gp_edges: Path,
    output_dir: Path
    ) -> Tuple[Path, Path]:

    # Load the phenotypes with prevalence and new phenotypes from gene-phenotype edges
    df_phenotypes_with_prevalence = pd.read_csv(phenotypes_with_prevalence)
    initial_phenotypes = df_phenotypes_with_prevalence["phenotype"].unique()

    # I replaced the codes below with this
    cols = ["phenotype", "target"]
    all_phenotypes_clean_df = df_phenotypes_with_prevalence[cols].drop_duplicates().copy()
    all_phenotypes_clean_df = all_phenotypes_clean_df.rename(columns={"target": "HPO_ID"}).reset_index(drop = True)

    # df_new_phenotypes_from_gp_edges = pd.read_csv(new_phenotypes_from_gp_edges)

    # # --- Build the Disease-HP Map ---
    # disease_to_hp_map = {}

    # # --- Fetch HPO closures for each disease ---
    # print("Fetching related phenotype using closures from Monarch API...")
    # for disease in initial_phenotypes:
    #     mondo_id = df_phenotypes_with_prevalence[df_phenotypes_with_prevalence["phenotype"] == disease]["mondo_id"].values[0]
    #     hp_list = fetch_hpo_closure(mondo_id)
    #     disease_to_hp_map[disease] = hp_list

    # # --- Collect all unique HP terms from the closures ---
    # all_closure_hp_terms = set()
    # for phenotype in disease_to_hp_map:
    #     phenotype_hpo_hpo_list = disease_to_hp_map.get(phenotype, [])
    #     all_closure_hp_terms.update(phenotype_hpo_hpo_list)

    # # --- Create a DataFrame for all unique HP terms and fetch their names ---
    # related_by_closures_df = pd.DataFrame(list(all_closure_hp_terms), columns=["HPO_ID"])
    # related_by_closures_df["Phenotype_Name"] = related_by_closures_df["HPO_ID"].apply(fetch_hpo_names)
    # related_by_closures_df.drop_duplicates(subset = 'HPO_ID', keep = 'first', inplace = True)

    # # --- Fetch related phenotypes using disease-phenotype associations from Monarch API ---
    # print("Fetching related phenotypes using disease-phenotype associations from Monarch API...")
    # related_by_associations = []
    # for disease in initial_phenotypes:
    #     mondo_id = df_phenotypes_with_prevalence[df_phenotypes_with_prevalence["phenotype"] == disease]["mondo_id"].values[0]
    #     related_phenotypes = fetch_related_phenotypes(mondo_id)
    #     phenotype_list = disease_to_hp_map[disease]
    #     phenotype_list.extend([item['HPO_ID'] for item in related_phenotypes])

    #     # Incase the main disease phenotype is missing for the list
    #     MAIN_HPO_ID = df_phenotypes_with_prevalence[df_phenotypes_with_prevalence["mondo_id"] == mondo_id]["target"].values[0]
    #     if MAIN_HPO_ID not in phenotype_list:
    #         phenotype_list.append(MAIN_HPO_ID)

    #     # Update the disease_to_hp_map with the new list of phenotypes, ensuring no duplicates
    #     disease_to_hp_map[disease] = list(set(phenotype_list))  # update HPO list and remove duplicates
    #     related_by_associations.extend(related_phenotypes)
    # related_by_associations_df = pd.DataFrame(related_by_associations)

    # # --- Combine the closures with new phenotypes from gene-phenotype edges ---
    # all_phenotypes_df = pd.concat([related_by_closures_df, related_by_associations_df, df_new_phenotypes_from_gp_edges[['HPO_ID', 'Phenotype_Name']]], ignore_index=True)
    # all_phenotypes_clean_df = all_phenotypes_df.drop_duplicates(subset = "HPO_ID", keep = "first")
    # all_phenotypes_clean_df = all_phenotypes_clean_df[all_phenotypes_clean_df["HPO_ID"] != "HP:0000001"] #Remove "All"
    
    # -- Add binary columns for each disease indicating if the HPO term is in the closure of that disease ---
    # for disease in initial_phenotypes:
    #     phenotype_col_name = f"Is_Primary_{disease}"
    #     all_phenotypes_clean_df[phenotype_col_name] = all_phenotypes_clean_df["HPO_ID"].apply(lambda x: 1 if x in disease_to_hp_map[disease] else 0)

    for disease in initial_phenotypes:
        phenotype_col_name = f"Is_Primary_{disease}"
        all_phenotypes_clean_df[phenotype_col_name] = (
            all_phenotypes_clean_df["phenotype"] == disease
        ).astype(int)         # Creates 1 if true, 0 if false for the whole column instantly

        # all_phenotypes_clean_df[phenotype_col_name] = all_phenotypes_clean_df["HPO_ID"].apply(lambda x: 1 if x in disease_to_hp_map[disease] else 0)
    print(all_phenotypes_clean_df)

    # Create the Ancestry Nodes
    ancestries = ['EAS', 'OCE', 'SAS', 'EUR', 'AMR', 'GME', 'AFR']
    ancestry_nodes = pd.DataFrame({
        'node_id': ancestries,
        # 'node_type': 'Ancestry',
        'ancestry_label': ['East Asian, Southeast Asian',
                        # 'Southeast Asia',
                        'Oceania',
                        'South Asian', 
                        'Central Europe, Eastern Europe, Central Asia',
                        'Latin America (Hispanic) and Caribbean',
                        'North Africa and Middle East',
                        'Sub-Saharan African',
                        ]
    })

    # Map your existing edge table (image 3) to these clean IDs
    # We extract the part after the '_'
    df_phenotypes_with_prevalence['ancestry'] = df_phenotypes_with_prevalence['source'].apply(map_to_anc_id)
    ancestry_phenotype_edges = df_phenotypes_with_prevalence[['source', 'target', 'prevalence', 'ancestry']].copy()

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    phenotype_features_dir = output_dir / "phenotype_features.csv"
    ancestry_nodes_dir = output_dir / "ancestry_nodes.csv"
    ancestry_phenotype_edges_dir = output_dir / "ancestry_phenotype_edges.csv"

    all_phenotypes_clean_df.to_csv(phenotype_features_dir, index=False)
    ancestry_nodes.to_csv(ancestry_nodes_dir, index=False)
    ancestry_phenotype_edges.to_csv(ancestry_phenotype_edges_dir, index=False)

    return phenotype_features_dir, ancestry_nodes_dir, ancestry_phenotype_edges_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Get phenotype features")
    parser.add_argument("--phenotypes-with-prevalence", default="phenotypes_with_prevalence.csv", help="Path to the phenotypes with prevalence CSV")
    parser.add_argument("--new-phenotypes-from-gp-edges", default="new_phenotypes_from_gp_edges.csv", help="Path to the new phenotypes from gene-phenotype edges CSV")
    parser.add_argument("--output-dir", default="output/gene_phenotype", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_phenotype_features(
        phenotypes_with_prevalence=Path(args.phenotypes_with_prevalence),
        new_phenotypes_from_gp_edges=Path(args.new_phenotypes_from_gp_edges),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the phenotype features data to {output_paths[0]}")
    print(f"Wrote the ancestry nodes to {output_paths[1]}")
    print(f"Wrote the ancestry-phenotype edges to {output_paths[2]}.")


if __name__ == "__main__":
    main()
