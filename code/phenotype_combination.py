import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tools.tools import (batch_fetch_hpo_names, 
                        fetch_related_phenotypes, 
                        batch_get_cui_safe,
                        batch_get_hpo_from_cuis,
                        )   


def load_umls_api_key() -> str:
    load_dotenv()
    umls_api_key = os.getenv("UMLS_API_KEY")
    if not umls_api_key:
        raise ValueError("UMLS_API_KEY is not set in the environment.")
    return umls_api_key



def combined_clinical_outcomes_data(
    phenotype_features: Path,
    phenotypes_with_prevalence: Path,
    side_effect_names_and_codes: Path,
    drug_treats_phenotype: Path,
    drug_causes_phenotype: Path,
    ancestry_phenotype_edges: Path,
    gene_phenotype_edges: Path,
    tissue_phenotype_edges: Path,
    phenotype_phenotype_edges_lin: Path,
    phenotype_phenotype_edges_ldsc: Path,
    variant_phenotype_edges: Path,
    output_dir: Path,
    ):

    # Phenotypes from core phenotypes
    df_phenotypes_core = pd.read_csv(phenotype_features)

    # Phenotypes from prevalence
    df_phenotypes_prevelence = pd.read_csv(phenotypes_with_prevalence)

    # All side effects
    df_adverse_effects_all = pd.read_csv(side_effect_names_and_codes)

    # Phenotypes from drug-treats-phenotypes
    df_drug_treats_phenotypes = pd.read_csv(drug_treats_phenotype)

    # Phenotypes from drug-causes-phenotypes
    df_drug_causes_phenotypes = pd.read_csv(drug_causes_phenotype)
    df_drug_causes_phenotypes = df_drug_causes_phenotypes[~df_drug_causes_phenotypes["meddraCode"].isna()]
    df_drug_causes_phenotypes["meddraCode"] = df_drug_causes_phenotypes["meddraCode"].astype(int).astype(str)

    # Phenotypes from ancestry-phenotype edges
    df_ancestry_phenotype_edges = pd.read_csv(ancestry_phenotype_edges)

    # Phenotypes from gene-phenotype edges
    df_gene_phenotype_edges = pd.read_csv(gene_phenotype_edges)

    # Phenotype-Phenotype edges (Lin)
    df_phenotype_phenotype_edges = pd.read_csv(phenotype_phenotype_edges_lin)

    # Phenotype-Phenotye edges (LDSC)
    df_phenotype_phenotype_edges_ldsc = pd.read_csv(phenotype_phenotype_edges_ldsc) 

    # Phenotypes from tissue-phenotype edges
    df_tissue_phenotype_edges = pd.read_csv(tissue_phenotype_edges)

    #Variant-phenotype edges
    df_variant_phenotype_edges = pd.read_csv(variant_phenotype_edges)

    # Get othe unique phenotypes from the drug-treats-phenotypes data
    df_other_phenotypes = df_drug_treats_phenotypes.groupby("target").first().reset_index().drop("source", axis = 1)

    # Combine the core phenotypes and other phenotypes
    df_core_and_other_phenotypes = pd.merge(df_phenotypes_core, df_other_phenotypes, left_on = "HPO_ID", right_on = "target", how = "outer")
    df_core_and_other_phenotypes["HPO_ID"] = df_core_and_other_phenotypes["HPO_ID"].fillna(df_core_and_other_phenotypes["target"])
    df_core_and_other_phenotypes.drop(columns="target", inplace = True)

    # Check for missing phenotype names but with an HPO ID in the combined dataframe
    missing_mask = df_core_and_other_phenotypes["phenotype"].isna() & df_core_and_other_phenotypes["HPO_ID"].notna()
    unique_missing_ids = df_core_and_other_phenotypes.loc[missing_mask, "HPO_ID"].unique()

    # Fetch the names for the missing HPO IDs using the batch_fetch_hpo_names function
    id_to_name_map = batch_fetch_hpo_names(unique_missing_ids, max_workers=10)

    # Update the phenotype names in the combined dataframe for the missing entries
    df_core_and_other_phenotypes.loc[missing_mask, "phenotype"] = (
        df_core_and_other_phenotypes.loc[missing_mask, "HPO_ID"].map(id_to_name_map)
    )

    # ---  Initial run to get CUIs for side effects and phenotypes ---
    # 1. Fetch unique IDs from source columns
    unique_meddra = df_adverse_effects_all["meddraCode"].dropna().unique().astype(str)
    unique_hpo = df_core_and_other_phenotypes["HPO_ID"].dropna().unique().astype(str)

    # 2. Run batch retrieval
    meddra_cui_map = batch_get_cui_safe(unique_meddra, "MDR", max_workers=5)
    hpo_cui_map = batch_get_cui_safe(unique_hpo, "HPO", max_workers=5)

    # 3. Initialize & populate "cui" column using .map()
    df_adverse_effects_all["cui"] = (
        df_adverse_effects_all["meddraCode"].astype(str).map(meddra_cui_map)
    )
    df_core_and_other_phenotypes["cui"] = (
        df_core_and_other_phenotypes["HPO_ID"].astype(str).map(hpo_cui_map)
    )

    # 4. Handle missing CUIs for side effects by searching by event name
    meddra_missing_mask = df_adverse_effects_all["cui"].isna() | (df_adverse_effects_all["cui"] == "None")
    missing_events = df_adverse_effects_all.loc[meddra_missing_mask, "event"].dropna().unique()

    if len(missing_events) > 0:
        print(f"Running fallback search for {len(missing_events)} unmapped event names...")
        # input_type="atom" searches by exact text string rather than source code
        event_cui_map = batch_get_cui_safe(missing_events, "MDR", input_type=None, max_workers=5)
        
        df_adverse_effects_all.loc[meddra_missing_mask, "cui"] = (
            df_adverse_effects_all.loc[meddra_missing_mask, "event"].map(event_cui_map)
        )

    # 5. Fill missing side effect and phenotype CUIs with the MedDRA code and HPO IDS if CUI not available available
    df_adverse_effects_all["cui"] = df_adverse_effects_all["cui"].fillna(df_adverse_effects_all["meddraCode"])
    df_adverse_effects_all["meddraCode"] = df_adverse_effects_all["meddraCode"].astype(str)
    df_core_and_other_phenotypes["cui"] = df_core_and_other_phenotypes["cui"].fillna(df_core_and_other_phenotypes["HPO_ID"])
    # ------------------------------------------------------------------ 

    # Add a new column to indicate whether the row is a side effect or a phenotype
    df_adverse_effects_all["is_side_effect"] = 1
    df_core_and_other_phenotypes["is_phenotype"] = 1

    # Combine the two dataframes on the "cui" column using an outer join
    clinical_outcomes = pd.merge(df_core_and_other_phenotypes, df_adverse_effects_all, on = "cui", how = "outer")

    # Fill the "outcome" column with the "phenotype" values where available, otherwise use the "event" values
    clinical_outcomes["outcome"] = clinical_outcomes["phenotype"].fillna(clinical_outcomes["event"])
    clinical_outcomes.drop(columns = ["phenotype", "event"], inplace = True)

    # Fill NaN values in the "is_phenotype" and "is_side_effect" columns with 0
    clinical_outcomes["is_phenotype"] = clinical_outcomes["is_phenotype"].apply(lambda x: 0 if pd.isnull(x) else x)
    clinical_outcomes["is_side_effect"] = clinical_outcomes["is_side_effect"].apply(lambda x: 0 if pd.isnull(x) else x)

    # Reorder the columns for clarity
    clinical_outcomes = clinical_outcomes[["cui", "HPO_ID", "meddraCode", "outcome", "Is_Primary_T2D", "Is_Primary_Obesity", 
                                       "Is_Primary_Hypertension", "Is_Primary_Hyperglycemia", "is_phenotype", "is_side_effect", 
                                       "count", "llr"]]

    ############# First set of mappings to get canonical CUIs for MedDRA and HPO IDs #############
    hpo_to_cui_all_df = clinical_outcomes[~clinical_outcomes["HPO_ID"].isna()][["HPO_ID", "cui"]]
    hpo_to_cui_all_dict = hpo_to_cui_all_df.set_index('HPO_ID')['cui'].to_dict()

    meddra_to_cui_all_df = clinical_outcomes[~clinical_outcomes["meddraCode"].isna()][["meddraCode", "cui"]]
    meddra_to_cui_all_dict = meddra_to_cui_all_df.set_index('meddraCode')['cui'].to_dict()

    clinical_outcomes = clinical_outcomes.sort_values(
        by = ["cui", "llr"], ascending = [True, False]).drop_duplicates(
        subset = ["cui"], keep = "first"
    )
    ##############################################################################################

    # Identify rows where HPO_ID is missing but a CUI exists
    missing_hpo_mask = clinical_outcomes["HPO_ID"].isna() & clinical_outcomes["cui"].str.startswith("C")
    unique_cuis_to_fetch = clinical_outcomes.loc[missing_hpo_mask, "cui"].unique().tolist()

    if len(unique_cuis_to_fetch) > 0:
        print(f"Found {len(unique_cuis_to_fetch)} unique CUIs to query...")
        
        # Run multithreaded fetch (max_workers=5 keeps requests at ~10 req/sec)
        # UMLS_API_KEY = load_umls_api_key()
        cui_to_hpo_map = batch_get_hpo_from_cuis(
            unique_cuis_to_fetch,
            # api_key=UMLS_API_KEY,
            max_workers=5,
        )
        
        # Update missing HPO_IDs in clinical_outcomes
        clinical_outcomes.loc[missing_hpo_mask, "HPO_ID"] = (
            clinical_outcomes.loc[missing_hpo_mask, "cui"].map(cui_to_hpo_map)
        )

    # -- Create a combined phenotype features table with their relationships to the core phenotypes -- #
    initial_phenotypes = df_phenotypes_core["phenotype"].tolist()
    disease_to_hp_map = {}
    for disease in initial_phenotypes:
        hpo_id = df_phenotypes_prevelence[df_phenotypes_prevelence["phenotype"] == disease]["target"].values[0]
        mondo_id = df_phenotypes_prevelence[df_phenotypes_prevelence["phenotype"] == disease]["mondo_id"].values[0]
        related_phenotypes = fetch_related_phenotypes(mondo_id)
        phenotype_list = [item['HPO_ID'] for item in related_phenotypes]
        phenotype_list.append(hpo_id)
        disease_to_hp_map[disease] = list(set(phenotype_list))

    for disease in initial_phenotypes:
        phenotype_col_name = f"Is_Primary_{disease}"
        target_hpos = set(disease_to_hp_map.get(disease, []))
        
        # Assign 1 if HPO_ID matches any target HPO ID for the disease, otherwise 0
        clinical_outcomes[phenotype_col_name] = clinical_outcomes["HPO_ID"].apply(
            lambda x: 1 if pd.notna(x) and x in target_hpos else 0
        )

    clinical_outcomes_final = clinical_outcomes.copy()
    clinical_outcomes_final["has_side_effect_stats"] = 1
    side_effect_stats_mask = clinical_outcomes_final["count"].isna() & clinical_outcomes_final["llr"].isna()
    clinical_outcomes_final.loc[side_effect_stats_mask , "has_side_effect_stats"] = 0
    clinical_outcomes_final["count"] = clinical_outcomes_final["count"].fillna(0)
    clinical_outcomes_final["llr"] = clinical_outcomes_final["llr"].fillna(0)


    clinical_outcomes_final["is_core_phenotype"] = 0
    core_flag_mask = clinical_outcomes_final["outcome"].isin(initial_phenotypes)
    clinical_outcomes_final.loc[core_flag_mask, "is_core_phenotype"] = 1
    clinical_outcomes_final = clinical_outcomes_final[["cui", "HPO_ID", "meddraCode", "outcome", "is_core_phenotype", "Is_Primary_T2D", 
                                                    "Is_Primary_Obesity", "Is_Primary_Hypertension", "Is_Primary_Hyperglycemia", 
                                                    "is_phenotype", "is_side_effect", "count", "llr", "has_side_effect_stats"]]


    # Ensure the output directory exists# Define aggregation strategies per column type
    aggregation_rules = {
        # Keep non-null IDs/Codes
        'cui': 'first',
        'meddraCode': 'first',
        'outcome': 'first',  # Keeps primary label (e.g., 'autoimmune disorder')
        
        # Binary indicator flags: take maximum (if 1 in any row, result is 1)
        'is_core_phenotype': 'max',
        'Is_Primary_T2D': 'max',
        'Is_Primary_Obesity': 'max',
        'Is_Primary_Hypertension': 'max',
        'Is_Primary_Hyperglycemia': 'max',
        'is_phenotype': 'max',
        'is_side_effect': 'max',
        'has_side_effect_stats': 'max',
        
        # Quantitative stats: take max or sum (side effect stats override 0/NaN)
        'count': 'max',
        'llr': 'max'
    }

    # Create a temporary grouping key: use HPO_ID if present, otherwise fall back to CUI
    clinical_outcomes_final['HPO_ID'] = clinical_outcomes_final['HPO_ID'].fillna(clinical_outcomes_final['cui'])
    clinical_outcomes_final['meddraCode'] = clinical_outcomes_final['meddraCode'].fillna(clinical_outcomes_final['cui'])

    # Group by the composite key using your aggregation rules
    clinical_outcomes_final_comp = (
        clinical_outcomes_final
        .groupby('HPO_ID', as_index=False)
        .agg(aggregation_rules)
    )
    

    ############# Second set of mappings to get canonical CUIs for MedDRA and HPO IDs #############
    hpo_to_cui_df = clinical_outcomes_final_comp[~clinical_outcomes_final_comp["HPO_ID"].isna()][["HPO_ID", "cui"]]
    hpo_to_cui_dict = hpo_to_cui_df.set_index('HPO_ID')['cui'].to_dict()

    meddra_to_cui_df = clinical_outcomes_final_comp[~clinical_outcomes_final_comp["meddraCode"].isna()][["meddraCode", "cui"]]
    meddra_to_cui_dict = meddra_to_cui_df.set_index('meddraCode')['cui'].to_dict()

    # Copy un-aggregated df to map each meddraCode to its canonical CUI
    meddra_hpo_df = clinical_outcomes_final[['meddraCode', 'HPO_ID']].dropna(subset=['meddraCode']).copy()

    # Map via HPO_ID to get the canonical CUI; fall back to original MedDRA if unmapped
    meddra_hpo_df['canonical_cui'] = meddra_hpo_df['HPO_ID'].map(hpo_to_cui_dict)
    meddra_hpo_df_found = meddra_hpo_df[~meddra_hpo_df["canonical_cui"].isna()]

    # Build the final MedDRA -> Canonical CUI dictionary
    meddra_to_canonical_cui_dict = (
        meddra_hpo_df_found
        .set_index('meddraCode')['canonical_cui']
        .to_dict()
    )
    ###############################################################################################

    # core phenotypes
    df_phenotypes_core["cui"] = df_phenotypes_core["HPO_ID"].map(hpo_to_cui_dict)
    df_phenotypes_core = df_phenotypes_core[["cui", "HPO_ID", "phenotype", "Is_Primary_T2D", "Is_Primary_Obesity",
                                            "Is_Primary_Hypertension", "Is_Primary_Hyperglycemia"]]
    
    # ancestry phenotype edge
    df_ancestry_phenotype_edges["cui"] = df_ancestry_phenotype_edges["target"].map(hpo_to_cui_dict)

    # Phenotypes from drug-treats-phenotypes
    df_drug_treats_phenotypes["cui"] = df_drug_treats_phenotypes["target"].map(hpo_to_cui_dict)
    df_drug_treats_phenotypes["cui"] = df_drug_treats_phenotypes["target"].map(hpo_to_cui_all_dict)

    # drug causes side effects
    df_drug_causes_phenotypes["cui"] = df_drug_causes_phenotypes["meddraCode"].map(meddra_to_cui_dict)
    df_drug_causes_phenotypes["cui"] = df_drug_causes_phenotypes["meddraCode"].map(meddra_to_canonical_cui_dict)
    df_drug_causes_phenotypes["cui"] = df_drug_causes_phenotypes["meddraCode"].map(meddra_to_cui_all_dict)

    # Gene - Phenotype edges
    df_gene_phenotype_edges["cui"] = df_gene_phenotype_edges["target"].map(hpo_to_cui_dict)

    # Phenotype - Phenotype edges (Lin)
    df_phenotype_phenotype_edges["source_cui"] = df_phenotype_phenotype_edges["source_hpo"].map(hpo_to_cui_dict)
    df_phenotype_phenotype_edges["source_cui"] = df_phenotype_phenotype_edges["source_hpo"].map(hpo_to_cui_all_dict)
    df_phenotype_phenotype_edges["target_cui"] = df_phenotype_phenotype_edges["target_hpo"].map(hpo_to_cui_dict)
    df_phenotype_phenotype_edges["target_cui"] = df_phenotype_phenotype_edges["target_hpo"].map(hpo_to_cui_all_dict)

    # Phenotype - Phenotype edges (LDSC)
    df_phenotype_phenotype_edges_ldsc["source_cui"] = df_phenotype_phenotype_edges_ldsc["source"].map(hpo_to_cui_dict)
    df_phenotype_phenotype_edges_ldsc["target_cui"] = df_phenotype_phenotype_edges_ldsc["target"].map(hpo_to_cui_dict)
   
    # Tissue - Phenotype edges
    df_tissue_phenotype_edges["cui"] = df_tissue_phenotype_edges["target_hpo"].map(hpo_to_cui_dict)
    df_tissue_phenotype_edges["cui"] = df_tissue_phenotype_edges["target_hpo"].map(hpo_to_cui_all_dict)

    # Variant - Phenotype edges
    df_variant_phenotype_edges["cui"] = df_variant_phenotype_edges["target_phenotype"].map(hpo_to_cui_dict)

    # Save the tissue_phenotype edges data to CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    df_phenotypes_core_dir = output_dir / "phenotype_features_cui.csv"
    df_ancestry_phenotype_edges_dir = output_dir / "ancestry_phenotype_edges_cui.csv"
    df_drug_treats_phenotypes_dir = output_dir / "drug_treats_phenotype_edges_cui.csv"
    df_drug_causes_phenotypes_dir = output_dir / "drug_causes_phenotype_edges_cui.csv"
    df_gene_phenotype_edges_dir = output_dir / "gene_phenotype_edges_cui.csv"
    df_phenotype_phenotype_edges_dir = output_dir / "phenotype_phenotype_edges_lin_cui.csv"
    df_phenotype_phenotype_edges_ldsc_dir = output_dir / "phenotype_phenotype_edges_ldsc_cui.csv"
    df_tissue_phenotype_edges_dir = output_dir / "tissue_phenotype_edges_cui.csv"
    df_variant_phenotype_edges_dir = output_dir / "variant_phenotype_edges_cui.csv"
    clinical_outcomes_final_dir = output_dir / "clinical_outcomes_final_cui.csv"

    # Save the phenotye_phenotype edges data to CSV files
    df_phenotypes_core.to_csv(df_phenotypes_core_dir, index=False)
    df_ancestry_phenotype_edges.to_csv(df_ancestry_phenotype_edges_dir, index=False)
    df_drug_treats_phenotypes.to_csv(df_drug_treats_phenotypes_dir, index=False)
    df_drug_causes_phenotypes.to_csv(df_drug_causes_phenotypes_dir, index=False)
    df_gene_phenotype_edges.to_csv(df_gene_phenotype_edges_dir, index=False)
    df_phenotype_phenotype_edges.to_csv(df_phenotype_phenotype_edges_dir, index=False)
    df_phenotype_phenotype_edges_ldsc.to_csv(df_phenotype_phenotype_edges_ldsc_dir, index=False)
    df_tissue_phenotype_edges.to_csv(df_tissue_phenotype_edges_dir, index=False)
    df_variant_phenotype_edges.to_csv(df_variant_phenotype_edges_dir, index=False)
    clinical_outcomes_final_comp.to_csv(clinical_outcomes_final_dir, index=False)  

    return (
        df_phenotypes_core_dir,
        df_ancestry_phenotype_edges_dir,
        df_drug_treats_phenotypes_dir,
        df_drug_causes_phenotypes_dir,
        df_gene_phenotype_edges_dir,
        df_phenotype_phenotype_edges_dir,
        df_phenotype_phenotype_edges_ldsc_dir,
        df_tissue_phenotype_edges_dir,
        df_variant_phenotype_edges_dir,
        clinical_outcomes_final_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Get phenotype features")
    parser.add_argument("--phenotype-features", default="phenotype_features.csv", help="Path to the phenotype features CSV")
    parser.add_argument("--phenotypes-with-prevalence", default="phenotypes_with_prevalence.csv", help="Path to the phenotypes with prevalence CSV")
    parser.add_argument("--side-effect-names-and-codes", default="side_effect_names_and_codes.csv", help="Path to the side effect names and codes CSV")
    parser.add_argument("--drug-causes-phenotype", default="drug_causes_phenotype_edges.csv", help="Path to the drug-causes-phenotype edges CSV")
    parser.add_argument("--drug-treats-phenotype", default="drug_treats_phenotype_edges.csv", help="Path to the drug-treats-phenotype edges CSV")
    parser.add_argument("--ancestry-phenotype-edges", default="ancestry_phenotype_edges.csv", help="Path to the ancestry-phenotype edges CSV")
    parser.add_argument("--gene-phenotype-edges", default="gene_phenotype_edges.csv", help="Path to the gene-phenotype edges CSV")
    parser.add_argument("--phenotype-phenotype-edges-lin", default="phenotype_phenotype_edges.csv", help="Path to the phenotype-phenotype edges (Lin) CSV")
    parser.add_argument("--phenotype-phenotype-edges-ldsc", default="phenotypes_phenotypes_ldsc.csv", help="Path to the phenotype-phenotype edges (LDSC) CSV")
    parser.add_argument("--tissue-phenotype-edges", default="tissue_phenotype_edges.csv", help="Path to the tissue-phenotype edges CSV")
    parser.add_argument("--variant-phenotype-edges", default="vp_edges_refined.csv", help="Path to the variant-phenotype edges CSV")
    parser.add_argument("--output-dir", default="output/gene_phenotype", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = combined_clinical_outcomes_data(
        phenotype_features=Path(args.phenotype_features),
        phenotypes_with_prevalence=Path(args.phenotypes_with_prevalence),
        side_effect_names_and_codes=Path(args.side_effect_names_and_codes),
        drug_causes_phenotype=Path(args.drug_causes_phenotype),
        drug_treats_phenotype=Path(args.drug_treats_phenotype),
        ancestry_phenotype_edges=Path(args.ancestry_phenotype_edges),
        gene_phenotype_edges=Path(args.gene_phenotype_edges),
        phenotype_phenotype_edges_lin=Path(args.phenotype_phenotype_edges_lin),
        phenotype_phenotype_edges_ldsc=Path(args.phenotype_phenotype_edges_ldsc),
        tissue_phenotype_edges=Path(args.tissue_phenotype_edges),
        variant_phenotype_edges=Path(args.variant_phenotype_edges),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the mapped core phenotypes data to {output_paths[0]}")
    print(f"Wrote the mapped ancestry-phenotype edges data to {output_paths[1]}")
    print(f"Wrote the mapped drug-treats-phenotype edges data to {output_paths[2]}")
    print(f"Wrote the mapped drug-causes-phenotype edges data to {output_paths[3]}")
    print(f"Wrote the mapped gene-phenotype edges data to {output_paths[4]}")
    print(f"Wrote the mapped phenotype-phenotype edges (Lin) data to {output_paths[5]}")
    print(f"Wrote the mapped phenotype-phenotype edges (LDSC) data to {output_paths[6]}")
    print(f"Wrote the mapped tissue-phenotype edges data to {output_paths[7]}")
    print(f"Wrote the mapped variant-phenotype edges data to {output_paths[8]}")
    print(f"Wrote the mapped clinical outcomes data to {output_paths[9]}")

if __name__ == "__main__":
    main()