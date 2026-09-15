import os
import urllib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pronto import Ontology

def get_hpo_obo():
    obo_path = "data/obo/hp.obo"
    hpo_url = "http://purl.obolibrary.org/obo/hp.obo"

    # Automatically download hp.obo if it doesn't exist locally
    if not os.path.exists(obo_path):
        print(f"Downloading the latest {obo_path} from OBO Foundry...")
        # Using a standard User-Agent to prevent server blocking
        req = urllib.request.Request(hpo_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(obo_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete!")

    # Load into pronto
    print("Loading HPO ontology into memory...")
    onto = Ontology(obo_path)
    return onto


def get_intrinsic_ic(term_id, TOTAL_TERMS, descendant_counts):
    """
    Calculates Intrinsic Information Content based on ontology topology.
    A leaf node with 0 sub-classes has high IC.
    A root node with thousands of sub-classes has low IC.
    """
    if term_id not in descendant_counts:
        return 0.0
    
    num_descendants = descendant_counts[term_id]
    # Intrinsic IC formula: -log10((descendants + 1) / total_terms)
    probability = (num_descendants + 1) / TOTAL_TERMS
    return -np.log10(probability)

def find_mica(term_a, term_b, onto, descendant_counts, total_terms):
    """Finds the shared ancestor with the highest Intrinsic Information Content."""
    ancestors_a = set(onto[term_a].superclasses(with_self=True))
    ancestors_b = set(onto[term_b].superclasses(with_self=True))
    shared_ancestors = ancestors_a.intersection(ancestors_b)
    
    if not shared_ancestors:
        return None
    
    # Select the shared ancestor that has the maximum intrinsic IC
    mica = max(shared_ancestors, key=lambda term: get_intrinsic_ic(term.id, total_terms, descendant_counts))
    return mica.id

def calculate_intrinsic_lin(term_a, term_b, TOTAL_TERMS, onto, descendant_counts):
    """Computes the Lin Similarity Score bounded between 0 and 1 without external data."""
    if term_a == term_b:
        return 1.0
        
    ic_a = get_intrinsic_ic(term_a, TOTAL_TERMS, descendant_counts)
    ic_b = get_intrinsic_ic(term_b, TOTAL_TERMS, descendant_counts)
    
    # If either term is invalid or has no information, similarity is 0
    if ic_a == 0 or ic_b == 0:
        return 0.0
        
    mica_id = find_mica(term_a, term_b, onto, descendant_counts, TOTAL_TERMS)
    if not mica_id:
        return 0.0
        
    ic_mica = get_intrinsic_ic(mica_id, TOTAL_TERMS, descendant_counts)
    
    # Lin's formula using intrinsic IC
    return (2.0 * ic_mica) / (ic_a + ic_b)

def build_phenotype_phenotype_edges(
        phenotype_features: Path,
        output_dir: Path,
    ) -> Path:
    
    # Load the phenotype features
    df_phenotype_node = pd.read_csv(phenotype_features)
    target_phenotypes = df_phenotype_node['HPO_ID'].dropna().astype(str).tolist()

    #  Load HPO-OBO
    onto = get_hpo_obo()

    TOTAL_TERMS = len(onto)
    descendant_counts = {}

    print("Precomputing ontology structure...")
    for term in onto.terms():
        if term.id:
            # Count how many unique sub-classes (descendants) this term has
            descendants = set(term.subclasses(with_self=True))
            descendant_counts[term.id] = len(descendants)

    edge_records = []

    # Loop through all pairs to build the flat edge list
    for i in range(len(target_phenotypes)):
        for j in range(i + 1, len(target_phenotypes)): # i+1 avoids duplicates and self-loops
            p1 = target_phenotypes[i]
            p2 = target_phenotypes[j]

            score = calculate_intrinsic_lin(p1, p2, TOTAL_TERMS, onto, descendant_counts)
        
            # We only save edges with some semantic relationship to keep the graph crisp
            if score > 0.05: 
                edge_records.append({
                    "source_hpo": p1,
                    "target_hpo": p2,
                    "lin_similarity": round(score, 4)
            })

    df_phenotype_phenoype_edges = pd.DataFrame(edge_records)

    # Save the drug_phenotype (causes) edges data to CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    phenotype_phenotype_edges_dir = output_dir / "phenotype_phenotype_lin_cui_all.csv"

    # Save the phenotye_phenotype edges data to CSV files
    df_phenotype_phenoype_edges.to_csv(phenotype_phenotype_edges_dir, index=False)
    return phenotype_phenotype_edges_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Get phenotype-phenotype edge data")
    parser.add_argument("--phenotype-features", default="clinical_outcomes.csv", help="Path to the phenotype features CSV")
    parser.add_argument("--output-dir", default="output/clinical_outcomes_edges", help="Directory for the phenotype-phenotype edges outputs")
    args = parser.parse_args()

    output_path = build_phenotype_phenotype_edges(
        phenotype_features=Path(args.phenotype_features),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the phenotype-phenotype edges data to {output_path}")


if __name__ == "__main__":
    main()
