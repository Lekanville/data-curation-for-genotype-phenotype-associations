import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from chembl_webresource_client.new_client import new_client

def build_drug_phenotype_data(
        phenotype_features: Path,
        drug_indications: Path,
        phenotype_codes: Path,
        output_dir: Path,
    ) -> Tuple[Path, Path, Path, Path, Path]:
    """Builds the drug-phenotype edges data and saves it to CSV files."""

    # Load the phenotype features, drug indications and phenotype codes CSV files
    df_phenotype_node = pd.read_csv(phenotype_features)

    df_indications = pd.read_csv(drug_indications)
    df_indications.rename(columns ={"molecule_chembl_id":"chembl_id"}, inplace = True)

    df_phenotype_codes = pd.read_csv(phenotype_codes)
    initial_phenotypes = df_phenotype_codes["phenotype"].unique()
    core_map = {}
    for disease in initial_phenotypes:
        mesh_id = df_phenotype_codes[df_phenotype_codes["phenotype"] == disease]["mesh_id"].values[0]
        hpo_id = df_phenotype_codes[df_phenotype_codes["phenotype"] == disease]["target"].values[0]
        core_map[mesh_id] = hpo_id


    # -- PART 1: FOR DRUG_CAUSES_PHENOTYPE EDGES ---
    ot_ae_url = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.12/output/openfda_significant_adverse_drug_reactions/part-00000-7c26a734-4175-4391-9060-5de87b7d62ed-c000.snappy.parquet"
    df_adverse_effects = pd.read_parquet(ot_ae_url)

    full_drug_list = df_indications['chembl_id'].unique().tolist()
    df_drug_causes_phenotype_edges = df_adverse_effects[df_adverse_effects["chembl_id"].isin(full_drug_list)]

    # getting the edge features ofr teh dru causes phenotype
    df_drug_causes_phenotype_edges['llr'] = pd.to_numeric(df_drug_causes_phenotype_edges['llr'], errors='coerce')
    df_drug_causes_phenotype_edges['llr'] = df_drug_causes_phenotype_edges['llr'].fillna(0.0)
    df_drug_causes_phenotype_edges['llr_log'] = np.log10(df_drug_causes_phenotype_edges['llr'] + 1)
    scaler = MinMaxScaler()
    df_drug_causes_phenotype_edges['llr_norm'] = scaler.fit_transform(df_drug_causes_phenotype_edges[['llr_log']])

    # Geting the side effect features by calculating aggregated statistical features (e.g., mean count and llr) aligned with the mapping order
    meddra_ids = df_drug_causes_phenotype_edges['meddraCode'].unique().tolist()
    df_side_effects = (
        df_drug_causes_phenotype_edges.groupby('meddraCode')[['count', 'llr']]
        .mean()
        .reindex(meddra_ids) # Ensures the rows match the exact order of meddra_to_idx
    ).reset_index()

    df_side_effect_names = df_adverse_effects.groupby("meddraCode").first().reset_index().drop(["chembl_id", "count", "llr", "critval"], axis = 1)
    df_side_effects_codes = df_side_effects.copy()
    df_side_effects_codes["meddraCode"] = df_side_effects_codes["meddraCode"].astype("Int64").astype("str")
    df_side_effect_names_and_codes = pd.merge(df_side_effect_names, df_side_effects_codes, on="meddraCode", how = "inner")

    
    # -- PART 2: FOR DRUG_TREATS_PHENOTYPE EDGES ---
    df_edges = df_indications.copy()

    # 1. For HPOIDs that are already present in the 'efo_id' column, we can directly use them.
    # Get the column with HP IDs
    df_edges['final_hpo'] = df_edges['efo_id'].apply(lambda x: x if str(x).startswith('HP:') else None)

    # Apply the manual map if final_hpo is still empty
    df_edges['final_hpo'] = df_edges['final_hpo'].fillna(df_edges['mesh_id'].map(core_map))


    # a. Filter: Only keep rows where we successfully found an HPO ID
    treats_edges = df_edges[df_edges['final_hpo'].notna()].copy()

    # b. Filter: Only keep HPO IDs that exist in phenotype nodes
    valid_hpo_ids = set(df_phenotype_node['HPO_ID'])
    print(f"Valid HPO IDs in phenotype nodes: {len(valid_hpo_ids)}")
    ##########################################################################################################
    # Checking the result of the drug-treats-phenotypes without filtering for valid HPO IDs in the phenotype nodes
    # treats_edges = treats_edges[treats_edges['final_hpo'].isin(valid_hpo_ids)]
    ##########################################################################################################

    # c. Format for GAT (Drug -> treats -> Phenotype)
    df_drug_pheno_edges = treats_edges[['chembl_id', 'final_hpo']].drop_duplicates()
    df_drug_pheno_edges.columns = ['source', 'target']

    print(f"Total Drug-Phenotype edges for HPO IDS in the EFO column: {len(df_drug_pheno_edges)}")
    print(f"Unique Phenotypes targeted by HPO IDs in the EFO column: {df_drug_pheno_edges['target'].nunique()}")

    # --- PART 2: MONDO/ORPHANET BRIDGE ---
    # a. Load the Mondo to HPO mapping from the Monarch API
    hpo_url = "http://purl.obolibrary.org/obo/mondo/mappings/mondo.sssom.tsv"
    df_mondo = pd.read_csv(hpo_url, sep='\t', comment='#', low_memory=False)

    # b. Load HPO Annotations & Harmonize
    hpo_ann_url = "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
    df_hpo_ann = pd.read_csv(hpo_ann_url, sep='\t', comment='#', low_memory=False)
    df_hpo_ann['database_id'] = df_hpo_ann['database_id'].str.upper()


    # 1. Extract Mondo Bridge & Harmonize Bridge Prefixes
    df_mondo_bridge = df_mondo[
        df_mondo['object_id'].str.contains('OMIM|ORPHA', na=False, case=False)
    ][['subject_id', 'object_id']].rename(columns={
        'subject_id': 'mondo_id', 
        'object_id': 'database_id'
    })

    # CLEAN BRIDGE: Replace prefixes THEN upper
    df_mondo_bridge['database_id'] = df_mondo_bridge['database_id'].str.replace('Orphanet:', 'ORPHA:', case=False)
    df_mondo_bridge['database_id'] = df_mondo_bridge['database_id'].str.replace('OMIMPS:', 'OMIM:', case=False)
    df_mondo_bridge['database_id'] = df_mondo_bridge['database_id'].str.upper()
    df_mondo_bridge['mondo_id'] = df_mondo_bridge['mondo_id'].str.upper()

    # 2. Prepare Indications for Bridging
    df_needs_bridge = df_edges.copy()

    # CLEAN INDICATIONS: Replace prefixes THEN upper
    df_needs_bridge['efo_id'] = df_needs_bridge['efo_id'].str.replace('Orphanet:', 'ORPHA:', case=False)
    df_needs_bridge['efo_id'] = df_needs_bridge['efo_id'].str.upper()

    # PATH A: ChEMBL (MONDO) -> Mondo Bridge -> HPO Annotations
    df_path_a = pd.merge(df_needs_bridge, df_mondo_bridge, left_on='efo_id', right_on='mondo_id', how='inner')
    df_hpo_path_a = pd.merge(df_path_a, df_hpo_ann, on='database_id', how='inner')

    # PATH B: ChEMBL (Direct ORPHA/OMIM) -> HPO Annotations
    # This catches rows where efo_id is already an ORPHA or OMIM code
    df_hpo_path_b = pd.merge(df_needs_bridge, df_hpo_ann, left_on='efo_id', right_on='database_id', how='inner')

    # 5. Combine and Filter
    df_bridged_combined = pd.concat([df_hpo_path_a, df_hpo_path_b], ignore_index=True)

    ##########################################################################################################
    # Checking the result of the drug-treats-phenotypes without filtering for valid HPO IDs in the phenotype nodes
    # df_bridged_ids_with_hpo = (
    #     df_bridged_combined[df_bridged_combined['hpo_id'].isin(valid_hpo_ids)]
    #     [['chembl_id', 'hpo_id']]
    #     .drop_duplicates()
    #     .rename(columns={'chembl_id': 'source', 'hpo_id': 'target'})
    # )
    df_bridged_ids_with_hpo = (
        df_bridged_combined[['chembl_id', 'hpo_id']]
        .drop_duplicates()
        .rename(columns={'chembl_id': 'source', 'hpo_id': 'target'})
    )
    ##########################################################################################################

    # 6. Final Concatenation
    df_drug_treats_phenotype_edges = pd.concat([df_drug_pheno_edges, df_bridged_ids_with_hpo], ignore_index=True).drop_duplicates()

    print(f"Total Drug-Phenotype edges: {len(df_drug_treats_phenotype_edges)}")
    print(f"Unique Phenotypes targeted: {df_drug_treats_phenotype_edges['target'].nunique()}")


    # Save the drug_phenotype (causes) edges data to CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_causes_phenotype_edges_dir = output_dir / "drug_causes_phenotype_edges.csv"
    side_effects_dir = output_dir / "side_effects_features.csv"
    drug_treats_phenotype_edges_dir = output_dir / "drug_treats_phenotype_edges.csv"
    side_effect_names_and_codes_dir = output_dir / "side_effect_names_and_codes.csv"

    # Save the drug_phenotype (treats) edges data to CSV files
    df_drug_causes_phenotype_edges.to_csv(drug_causes_phenotype_edges_dir, index=False)
    df_side_effects.to_csv(side_effects_dir, index=False)
    df_side_effect_names_and_codes.to_csv(side_effect_names_and_codes_dir, index=False)
    df_drug_treats_phenotype_edges.to_csv(drug_treats_phenotype_edges_dir, index=False)

    return drug_causes_phenotype_edges_dir, drug_treats_phenotype_edges_dir

def main() -> None:
    parser = argparse.ArgumentParser(description="Get drug-phenotype data")
    parser.add_argument("--phenotype-features", default="phenotype_features.csv", help="Path to the phenotype features CSV")
    parser.add_argument("--drug-indications", default="drug_indications.csv", help="Path to the drug indications CSV")
    parser.add_argument("--phenotype-code-file", default="data/phenotypes_prevalence.csv", help="Path to the phenotype codes CSV")
    parser.add_argument("--output-dir", default="output/drug_phenotype_edges", help="Directory for the drug-phenotype edges outputs")
    args = parser.parse_args()

    output_paths = build_drug_phenotype_data(
        phenotype_features=Path(args.phenotype_features),
        drug_indications=Path(args.drug_indications),
        phenotype_codes=Path(args.phenotype_code_file),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the drug-causes-phenotype edges data to {output_paths[0]}")
    print(f"Wrote the drug-treats-phenotype edges data to  {output_paths[1]}")


if __name__ == "__main__":
    main()
