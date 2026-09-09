import io
import os
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from sklearn.preprocessing import MinMaxScaler

# def get_tissue_weights_for_phenotype(row):
    # hpo_to_folder = {
    #     "HP:0005978":"FUMA_public_job604009",   # FUMA-MAGMA for T2D
    #     "HP:0001513":"FUMA_public_job680502",   # Obesity
    #     "HP:0000822":"FUMA_public_job604009",   # Fallback Hypertension (use its own if available, or fallback)
    #     "HP:0003074":"FUMA_public_job604009",   # Fallback Hyperglycemia falls back to T2D (SAS)
    # }

    
    # active_cores = []
    # if row['Is_Primary_T2D'] == 1: active_cores.append("HP:0005978")
    # if row['Is_Primary_Obesity'] == 1: active_cores.append("HP:0001513")
    # if row['Is_Primary_Hypertension'] == 1: active_cores.append("HP:0000822")
    # if row['Is_Primary_Hyperglycemia'] == 1: active_cores.append("HP:0003074")
    
    # Map active cores to folders they actually have
    # available_folders = list(set([hpo_to_folder[c] for c in active_cores if c in hpo_to_folder]))
    
    
        
    # return available_folders # This list feeds your consolidation logic
    

def parse_tissue_enrichment_folder(raw_value: Any) -> Dict[str, str]:
    if raw_value is None:
        return {}

    if isinstance(raw_value, dict):
        return {str(k).strip(): str(v).strip() for k, v in raw_value.items()}

    if isinstance(raw_value, list):
        mapping: Dict[str, str] = {}
        for item in raw_value:
            if isinstance(item, str) and ':' in item:
                key, value = item.split(':', 1)
                mapping[key.strip().strip('"\'')]=value.strip().strip('"\'')
        return mapping

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if raw_value.startswith('{') and raw_value.endswith('}'):
            try:
                parsed = eval(raw_value)
                if isinstance(parsed, dict):
                    return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
            except Exception:
                pass

        if raw_value.startswith('[') and raw_value.endswith(']'):
            try:
                parsed = eval(raw_value)
                return parse_tissue_enrichment_folder(parsed)
            except Exception:
                pass

        if '::' in raw_value:
            mapping: Dict[str, str] = {}
            for part in raw_value.split('::'):
                if '-' in part:
                    key, value = part.split('-', 1)
                    mapping[key.strip()] = value.strip()
            return mapping

    return {}



def tissue_phenotype_edges(df_phenotypes, FUMA_BASE_DIR):
    print("=== Starting Master Tissue-Phenotype Loop ===")

    # Define your 10-tissue translation dictionary
    TISSUE_MAPPING = {
        "Adipose_Subcutaneous": "Adipose_Tissue",
        "Artery_Aorta": "Blood_Vessel",
        "Heart_Left_Ventricle": "Heart",
        "Kidney_Cortex": "Kidney",
        "Liver": "Liver",
        "Muscle_Skeletal": "Muscle",
        "Pancreas": "Pancreas",
        "Whole_Blood": "Blood",
        "Brain_Hypothalamus": "Brain",
        "Small_Intestine_Terminal_Ileum": "Small_Intestine"
    }
    
    TARGET_FILE_NAME = "magma_exp_gtex_v8_ts_general_avg_log2TPM.gsa.out"
    master_edge_records = []
    
    # Iterate dynamically over your 38 target phenotypes
    for _, row in df_phenotypes.iterrows():
        hpo_id = row['HPO_ID']
        phenotype_name = row['phenotype']
        mapped_folders = row['Available_FUMA_Folders']
        
        # Track the best p-value for each of our 10 core tissues across this phenotype's mapped folders
        tissue_p_values = {gtex_name: [] for gtex_name in TISSUE_MAPPING.keys()}
        
        for folder in mapped_folders:
            file_path = os.path.join(FUMA_BASE_DIR, folder, TARGET_FILE_NAME)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # Load whitespace-delimited MAGMA table
                df_magma = pd.read_csv(file_path, sep=r'\s+', comment='#')
                
                # Extract p-values for our specific 10 tissues
                for gtex_name, fuma_name in TISSUE_MAPPING.items():
                    magma_row = df_magma[df_magma['VARIABLE'] == fuma_name]
                    if not magma_row.empty:
                        p_val = float(magma_row['P'].values[0])
                        tissue_p_values[gtex_name].append(p_val)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

            
        # Consolidate: select the most significant p-value per tissue (Amalgamation Rule)
        for gtex_tissue, p_list in tissue_p_values.items():
            if not p_list:
                continue
                
            # Select minimum p-value (highest statistical signal)
            p_selected = min(p_list)
            
            # Continuous weight mapping calculation
            neg_log_p = -np.log10(max(p_selected, 1e-300))
            
            # Optional: You can filter for p < 0.05 here if you want to keep the graph sparse,
            # or remove this condition if your GAT layer expects a fully connected tissue density profile.
            # if p_selected < 0.05: 
            master_edge_records.append({
                "source_tissue": gtex_tissue,
                "target_hpo": hpo_id,
                "phenotype_name": phenotype_name,
                "p_value": p_selected,
                "weight": round(neg_log_p, 4)
            })

    # Save to your structured output folder
    df_master_edges = pd.DataFrame(master_edge_records)
    
    # Filter out insignificant associations first to keep the graph clean
    # df_master_edges = df_master_edges[df_master_edges['p_value'] < 0.05].copy()

    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Define a local grouping function to handle the 2D array requirement
    def scale_per_phenotype(group):
        weights = group[['weight']].values
        if weights.max() - weights.min() > 0:
            group['weight_norm'] = scaler.fit_transform(weights)
        else:
            group['weight_norm'] = 1.0 # Fallback if all weights are identical
        return group

    # Apply the scaler independently per phenotype group
    df_master_edges = df_master_edges.groupby('target_hpo').apply(
        scale_per_phenotype, 
        include_groups=False)
    df_master_edges['weight_norm'] = df_master_edges['weight_norm'].round(6)
    df_master_edges.reset_index(inplace = True)
    df_master_edges.drop(columns=["level_1"], inplace = True)

    print("=== SUCCESS ===")
    print(f"Compiled {len(df_master_edges)} total edges for your graph across all {len(df_phenotypes)}")
    return df_master_edges


def build_master_tissue_phenotype_edges(
        phenotype_features: Path,
        phenotypes_with_prevalence: Path,
        tissue_enrichment_file: Path,
        tissue_enrichment_map: Any,
        output_dir: Path,
    ) -> Path:

    df_phenotypes = pd.read_csv(phenotype_features)

    df_phenotypes_with_prevalence = pd.read_csv(phenotypes_with_prevalence)
    initial_phenotypes = df_phenotypes_with_prevalence["phenotype"].unique()

    enrichment_mapping = parse_tissue_enrichment_folder(tissue_enrichment_map)
    df_phenotypes['Available_FUMA_Folders'] = pd.Series([[] for _ in range(len(df_phenotypes))], index=df_phenotypes.index, dtype=object)

    print("=== Mapping the Tissue Enrichment Folders to the Phenotypes ===")
    for i in range(len(df_phenotypes)):
        active_cores = []
        for j in initial_phenotypes:
            HPO_ID = df_phenotypes_with_prevalence[df_phenotypes_with_prevalence["phenotype"] == j]["target"].values[0]
            check = f'Is_Primary_{j}'
            if df_phenotypes.loc[i, check] == 1:
                active_cores.append(HPO_ID)

        available_folders = [enrichment_mapping[c] for c in active_cores if c in enrichment_mapping]
        available_folders = list(dict.fromkeys(available_folders))

        df_phenotypes.at[i, 'Available_FUMA_Folders'] = available_folders

    print("=== Mapping Successful ===")


    FUMA_BASE_DIR = tissue_enrichment_file
    df_tissue_phenotype_edges = tissue_phenotype_edges(df_phenotypes, FUMA_BASE_DIR)

    # Save the tissue_phenotype edges data to CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue_phenotype_edges_dir = output_dir / "tissue_phenotype_edges.csv"

    # Save the phenotye_phenotype edges data to CSV files
    df_tissue_phenotype_edges.to_csv(tissue_phenotype_edges_dir, index=False)
    return tissue_phenotype_edges_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Get tissue-phenotype edge data")
    parser.add_argument("--phenotype-features", default="phenotype_features.csv", help="Path to the phenotype features CSV")
    parser.add_argument("--phenotypes-with-prevalence", default="phenotypes_with_prevalence.csv", help="Path to the phenotypes with prevalence CSV")
    parser.add_argument("--tissue-enrichment-file", help="Path to the enrichment folder(s)")
    parser.add_argument("--tissue-enrichment-map", help="tissue enrichment maps")
    parser.add_argument("--output-dir", default="output/tissue_phenotype_edges", help="Directory for the tissue-phenotype edges outputs")
    args = parser.parse_args()

    output_path = build_master_tissue_phenotype_edges(
        phenotype_features=Path(args.phenotype_features),
        phenotypes_with_prevalence=Path(args.phenotypes_with_prevalence),
        tissue_enrichment_file=Path(args.tissue_enrichment_file),
        tissue_enrichment_map=args.tissue_enrichment_map,
        output_dir=Path(args.output_dir)
    )

    print(f"Wrote the tissue-phenotype edges data to {output_path}")


if __name__ == "__main__":
    main()




# def parse_tissue_enrichment_folder(raw_value: Any) -> Dict[str, str]:
#     if raw_value is None:
#         return {}

#     if isinstance(raw_value, dict):
#         return {str(k).strip(): str(v).strip() for k, v in raw_value.items()}

#     if isinstance(raw_value, list):
#         mapping: Dict[str, str] = {}
#         for item in raw_value:
#             if isinstance(item, str) and ':' in item:
#                 key, value = item.split(':', 1)
#                 mapping[key.strip().strip('"\'')]=value.strip().strip('"\'')
#         return mapping

#     if isinstance(raw_value, str):
#         raw_value = raw_value.strip()
#         if raw_value.startswith('[') and raw_value.endswith(']'):
#             try:
#                 parsed = ast.literal_eval(raw_value)
#                 return parse_tissue_enrichment_folder(parsed)
#             except Exception:
#                 pass

#         if raw_value.startswith('{') and raw_value.endswith('}'):
#             try:
#                 parsed = ast.literal_eval(raw_value)
#                 if isinstance(parsed, dict):
#                     return parse_tissue_enrichment_folder(parsed)
#             except Exception:
#                 pass

#         mapping: Dict[str, str] = {}
#         for part in raw_value.split(','):
#             if ':' in part:
#                 key, value = part.split(':', 1)
#                 mapping[key.strip().strip('"\'')]=value.strip().strip('"\'')
#         return mapping

#     return {}