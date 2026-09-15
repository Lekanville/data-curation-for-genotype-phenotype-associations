import mygene
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def symbols_to_entrez(gene_symbols_ids):
    # 1b. Map Symbols to Entrez IDs
    print("Mapping Gene Symbols to Entrez IDs...")
    mapping_url = "http://mygene.info/v3/query"
    entrez_id_list = []

    for i in gene_symbols_ids:
        mapping_payload = {
            # 'q': ','.join(data_list), # Use 'q' parameter for query
            'q': i,                      # Use 'q' parameter for query
            'scopes': 'symbol',          # Target the symbol field
            'fields': '_id',             # Only need the Entrez ID
            'species': 'human',
            'dot_product': 'false',      # Ensure we get individual results
        }
        
        try:
            response_entrez = requests.post(mapping_url, data=mapping_payload)
            response_entrez.raise_for_status()
            mapping_data = response_entrez.json()
        
            for hit in mapping_data:
                gene_id = hit.get('_id')
                
                if gene_id and gene_id.isdigit():
                    entrez_id_list.append({"Gene_Symbol":i, "Entrez_ID":gene_id})
        
            
        except requests.exceptions.RequestException as err:
            print(f"❌ Error during symbol mapping: {err}")
            entrez_id_list.append({"Gene_Symbol":i, "Entrez_ID":""})

    df_entrez_from_symbols = pd.DataFrame(entrez_id_list)
    
    return (df_entrez_from_symbols)


def map_entrez_to_ensembl(entrez_ids):
    """
    Maps a list of Entrez IDs to Ensembl (Gencode) IDs.
    """
    mg = mygene.MyGeneInfo()
    
    # Query MyGene.info
    # We ask for the 'ensembl.gene' field
    result = mg.querymany(entrez_ids, 
                          scopes='entrezgene', 
                          fields='ensembl.gene', 
                          species='human')
    
    mapping_list = []
    for entry in result:
        entrez = entry.get('query')
        ensembl_data = entry.get('ensembl')
        
        # Ensembl data can be a list if a gene has multiple IDs
        if isinstance(ensembl_data, list):
            ensembl = ensembl_data[0].get('gene')
        elif isinstance(ensembl_data, dict):
            ensembl = ensembl_data.get('gene')
        else:
            ensembl = None
            
        if ensembl:
            mapping_list.append({'Entrez_ID': str(entrez), 'Ensembl_ID': ensembl})
            
    return pd.DataFrame(mapping_list)

def resolve_gtex_ids(ensembl_base_ids):
    """
    Maps base Ensembl IDs to the versioned GENCODE IDs used in GTEx v8.
    """
    url = "https://gtexportal.org/api/v2/reference/gene"
    # resolved_mapping = []
    resolved_mapping = {}
    
    # We chunk the requests to avoid API timeouts
    chunk_size = 50
    for i in range(0, len(ensembl_base_ids), chunk_size):
        chunk = ensembl_base_ids[i:i + chunk_size]
        params = {
            'geneId': chunk,
            'datasetId': 'gtex_v8'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json().get('data', [])
            
            for entry in data:
                # Map the base ID back to the versioned Gencode ID
                # Example: ENSG00000206527 -> ENSG00000206527.2
                # entrez_id = str(entry['entrezGeneId'])
                entrez_id = float(entry['entrezGeneId'])
                gencode_id = entry['gencodeId']
                # resolved_mapping.append({'Entrez_ID': entrez_id, 'Gencode_ID': gencode_id})
                resolved_mapping[entrez_id] = gencode_id
                
        except Exception as e:
            print(f"❌ Error resolving chunk: {e}")
            
    # return pd.DataFrame(resolved_mapping)
    return resolved_mapping

def gene_features_imputation(unique_gene_list: Path, vg_edges: Path, variant_tissue_edges: Path, gene_features: Path, vocab_size: int, output_dir: Path) -> Tuple[Path, Path]:
    """Impute missing gene features for the unique gene list."""

    # Load the unique gene list
    elite_genes = pd.read_csv(unique_gene_list)
    elite_genes['Entrez_ID'] = elite_genes['Entrez_ID'].apply(lambda x: str(x).split('.')[0])

    # Load the variant-gene edges and variant-tissue edges
    df_gene_from_modifiers = pd.read_csv(vg_edges)

    # Load the variant-gene-tissue edges
    df_gene_from_VGT = pd.read_csv(variant_tissue_edges)

    # Load the gene features
    df_gene_features = pd.read_csv(gene_features)
    print(f"Loaded {len(df_gene_features)} gene features.")

    # unique_from_modifiers
    unique_from_modifiers =  df_gene_from_modifiers.groupby(['Target_Gene_ID']).first().reset_index().drop(
      ["Source_Variant_rsid", "Edge_Feature_Impact"], axis = 1
    )

    # unique_from_VGT
    unique_from_VGT = df_gene_from_VGT.groupby(['Gene_ID']).first().reset_index().drop(
        ["rsid", "Variant_ID", "Tissue_ID", "P_Value",	"NES"], axis = 1
    )

    # elites_with_modifiers
    elites_with_modifiers = pd.merge(
        elite_genes, unique_from_modifiers, left_on = "Gene_Symbol", right_on = "Target_Gene_ID", how = "outer"
    )

    # elites_with_modifiers_and_VGT
    elites_with_modifiers_and_VGT = pd.merge(
        elites_with_modifiers, unique_from_VGT, left_on = "Gencode_ID", right_on = "Gene_ID", how = "outer"
    )

    # Combine the symbols
    elites_with_modifiers_and_VGT['Gene_Symbol'] = elites_with_modifiers_and_VGT['Gene_Symbol'].combine_first(
        elites_with_modifiers_and_VGT['Target_Gene_ID']
    )

    # Combine Ensemble IDs
    elites_with_modifiers_and_VGT['Ensembl_ID'] = elites_with_modifiers_and_VGT['Ensembl_ID_x'].combine_first(
        elites_with_modifiers_and_VGT['Ensembl_ID_y']
    )
    elites_with_modifiers_and_VGT.drop(["Target_Gene_ID", "Ensembl_ID_x", 'Ensembl_ID_y', 'Gene_ID'], axis = 1, inplace = True)

    #Gene from gene node features
    gene_with_features = df_gene_features["Gene_Symbol"].unique()

    # Identify genes without node features
    genes_without_node_features = elites_with_modifiers_and_VGT[~elites_with_modifiers_and_VGT['Gene_Symbol'].isin(gene_with_features)]['Gene_Symbol'].unique()
    print(f"Identified {len(genes_without_node_features)} genes without node features.")

    # Final Gene list from elites, modifiers, VGT and features list
    genes_node_with_ids = pd.merge(elites_with_modifiers_and_VGT, df_gene_features, on = "Gene_Symbol", how = "outer")

    # Combine Ensemble IDs
    genes_node_with_ids['Entrez_ID'] = genes_node_with_ids['Entrez_ID_x'].combine_first(
        genes_node_with_ids['Entrez_ID_y']
    )
    # genes_node_with_ids["Entrez_ID"] = genes_node_with_ids["Entrez_ID"].astype(float)
    genes_node_with_ids["Entrez_ID"] = pd.to_numeric(genes_node_with_ids["Entrez_ID"], errors='coerce')
    genes_node_with_ids.drop(["Entrez_ID_x", "Entrez_ID_y"], axis = 1, inplace = True)

    print(f"The final gene list with node features:\n{genes_node_with_ids.head()}")

    # Imputation for p_Li, gene length, molecular weight, PPI and expression specificity
    # 1. IDENTIFY NON-CODING VS CODING
    non_coding_mask = genes_node_with_ids['Gene_Symbol'].str.contains('LINC|MIR|LOC|SNOR|[-]', case=False, na=False)

    # 2. CALCULATE MEDIANS FROM "REAL" DATA ONLY
    median_pli        = genes_node_with_ids[genes_node_with_ids['pLI_Imputed_Flag'] == False]["pLI_Score"].median()
    median_geneLength = genes_node_with_ids[genes_node_with_ids['Gene_Length_Imputed_Flag'] == False]["Gene_Length"].median()
    median_molWeight  = genes_node_with_ids[genes_node_with_ids['M_W_Imputed_Flag'] == 0]["Molecular_Weight"].median()
    median_ppi        = genes_node_with_ids[genes_node_with_ids['PPI_Count_Imputed_Flag'] == 0]["PPI_Count"].median()
    median_exSp       = genes_node_with_ids[genes_node_with_ids['Ex_Sp_Flag_Imputed'] == 0]["Expression_Specificity_Score"].median()

    # FOR PROTEIN FEATURES (pLI, MW, and PPI)
    # 3. APPLY MEDIAN IMPUTATION FOR MISSING CODING DATA (Coding Only for Protein features)
    genes_node_with_ids.loc[~non_coding_mask & ((genes_node_with_ids['pLI_Imputed_Flag'] == True) | (genes_node_with_ids['pLI_Imputed_Flag'].isnull())), 'pLI_Score'] = median_pli
    genes_node_with_ids.loc[~non_coding_mask & ((genes_node_with_ids['M_W_Imputed_Flag'] == 1) | (genes_node_with_ids['M_W_Imputed_Flag'].isnull())), 'Molecular_Weight'] = median_molWeight
    genes_node_with_ids.loc[~non_coding_mask & ((genes_node_with_ids['PPI_Count_Imputed_Flag'] == 1) | (genes_node_with_ids['PPI_Count_Imputed_Flag'].isnull())), 'PPI_Count'] = median_ppi

    # 4. ENFORCE BIOLOGICAL ZERO FOR MISSING NON-CODING DATA
    genes_node_with_ids.loc[non_coding_mask & ((genes_node_with_ids['pLI_Imputed_Flag'] == True) | (genes_node_with_ids['pLI_Imputed_Flag'].isnull())), 'pLI_Score'] = 0.0
    genes_node_with_ids.loc[non_coding_mask & ((genes_node_with_ids['M_W_Imputed_Flag'] == 1) | (genes_node_with_ids['M_W_Imputed_Flag'].isnull())), 'Molecular_Weight'] = 0.0
    genes_node_with_ids.loc[non_coding_mask & ((genes_node_with_ids['PPI_Count_Imputed_Flag'] == 1) | (genes_node_with_ids['PPI_Count_Imputed_Flag'].isnull())), 'PPI_Count'] = 0.0

    # FOR STRUCTURAL FEATURES (Length and Specificity)
    # 5. APPLY MEDIAN IMPUTATION FOR MISSING STRUCTURAL FEATURES IN ALL TYPES OF DATA (CODING AND NON-CODING)
    genes_node_with_ids.loc[(genes_node_with_ids['Gene_Length_Imputed_Flag'] == True) | (genes_node_with_ids['Gene_Length_Imputed_Flag'].isnull()), 'Gene_Length'] = median_geneLength
    genes_node_with_ids.loc[(genes_node_with_ids['Ex_Sp_Flag_Imputed'] == 1) | (genes_node_with_ids['Ex_Sp_Flag_Imputed'].isnull()), 'Expression_Specificity_Score'] = median_exSp

    # 6. RECONCILE FLAGS (Fixed syntax errors)
    genes_node_with_ids.loc[genes_node_with_ids['pLI_Imputed_Flag'].isnull(), 'pLI_Imputed_Flag'] = True
    genes_node_with_ids.loc[genes_node_with_ids['Gene_Length_Imputed_Flag'].isnull(), 'Gene_Length_Imputed_Flag'] = True
    genes_node_with_ids.loc[genes_node_with_ids['M_W_Imputed_Flag'].isnull(), 'M_W_Imputed_Flag'] = 1
    genes_node_with_ids.loc[genes_node_with_ids['PPI_Count_Imputed_Flag'].isnull(), 'PPI_Count_Imputed_Flag'] = 1
    genes_node_with_ids.loc[genes_node_with_ids['Ex_Sp_Flag_Imputed'].isnull(), 'Ex_Sp_Flag_Imputed'] = 1


    # Imputation for Gene ontology terms
    # 1. Dynamically gather all GO column names
    go_bp_cols = [f'GO_BP_{i}' for i in range(vocab_size)]
    go_mf_cols = [f'GO_MF_{i}' for i in range(vocab_size)]
    go_cc_cols = [f'GO_CC_{i}' for i in range(vocab_size)]
    all_go_cols = go_bp_cols + go_mf_cols + go_cc_cols

    # 2. Identify which rows are missing GO data
    # We check if the first column (GO_BP_0) is NaN as a proxy for the whole set
    go_missing_mask = (genes_node_with_ids['GO_Data_Missing_Flag'] == 1) | (genes_node_with_ids['GO_BP_0'].isna())

    # 3. Bulk Impute with 0.0
    # This turns the "unknown" into a zero-vector, which is standard for GNNs
    genes_node_with_ids.loc[go_missing_mask, all_go_cols] = 0.0

    # 4. Finalise the Flag
    # Ensure every imputed row is marked with a 1 (or True)
    genes_node_with_ids.loc[go_missing_mask, 'GO_Data_Missing_Flag'] = 1


    # Rearrange the columns
    init = ["Gene_Symbol", "Entrez_ID", "Gencode_ID", "Ensembl_ID", "pLI_Score", "pLI_Imputed_Flag", "Gene_Length", 
        "Gene_Length_Imputed_Flag", "Molecular_Weight", "M_W_Imputed_Flag", "GO_Data_Missing_Flag"]
    end = ["PPI_Count_Imputed_Flag", "PPI_Count", "Expression_Specificity_Score", "Ex_Sp_Flag_Imputed"]

    all_cols = init + all_go_cols + end
    genes_node_full = genes_node_with_ids[all_cols]
    genes_node_full["Entrez_ID"] = genes_node_full["Entrez_ID"].astype(float)

    # Full imputed gene features
    print(f"Full imputed gene features:\n{genes_node_full.head()}")

    cleaned_symbols = genes_node_full[(genes_node_full['Entrez_ID'].isna())]["Gene_Symbol"].unique()
    len(cleaned_symbols)

    # Get entrez IDs for the cleaned symbols
    entrez_from_symbols = symbols_to_entrez(cleaned_symbols)
    if len(entrez_from_symbols) > 0:
        genes_node_full = pd.merge(genes_node_full, entrez_from_symbols, on = ["Gene_Symbol"], how = "outer")
        genes_node_full = genes_node_full.copy()
        genes_node_full['Entrez_ID'] = genes_node_full['Entrez_ID_x'].combine_first(genes_node_full['Entrez_ID_y'])
        genes_node_full.drop(["Entrez_ID_x", "Entrez_ID_y"], axis = 1, inplace = True)
        genes_node_full = genes_node_full[all_cols]

    cleaned_symbols = genes_node_full[(genes_node_full['Entrez_ID'].isna())]["Gene_Symbol"].unique()
    print(f"Number of cleaned symbols after entrez ID search: {len(cleaned_symbols)}")

    entrez_to_search = genes_node_full[(~genes_node_full['Entrez_ID'].isna()) & (genes_node_full['Gencode_ID'].isna())]['Entrez_ID'].unique()
    entrez_to_search = [str(i).split(".")[0] for i in entrez_to_search]
    print(f"Number of entrez IDs to search for Ensembl IDs: {len(entrez_to_search)}")

    # Map Entrez IDs to Ensembl IDs
    df_ensemble_entrez_mapping = map_entrez_to_ensembl(entrez_to_search)
    df_ensemble_entrez_mapping["Entrez_ID"] = df_ensemble_entrez_mapping["Entrez_ID"].astype(float)
    # ensembl_list = df_ensemble_entrez_mapping['Ensembl_ID'].tolist()

    mapped_entrez_ensemble = {}
    for i in range(len(df_ensemble_entrez_mapping)):
        entrez = df_ensemble_entrez_mapping.loc[i, "Entrez_ID"]
        ensemble = df_ensemble_entrez_mapping.loc[i, "Ensembl_ID"]
        mapped_entrez_ensemble[entrez] = ensemble
        
    genes_node_full["New_Ensembl_ID"] = genes_node_full["Entrez_ID"].map(mapped_entrez_ensemble).fillna("")  # or np.nan
    genes_node_full['Ensembl_ID'] = genes_node_full['Ensembl_ID'].combine_first(genes_node_full['New_Ensembl_ID'])
    genes_node_full.drop(["New_Ensembl_ID"], axis = 1, inplace = True)
    genes_node_full = genes_node_full[all_cols]

    # Get Ensembl IDs to search for Gencode IDs
    ensemble_to_search = genes_node_full[(~genes_node_full["Ensembl_ID"].isna()) & (genes_node_full['Gencode_ID'].isna())]["Ensembl_ID"].unique()
    gtex_id_map = resolve_gtex_ids(ensemble_to_search)

    if len(gtex_id_map) > 0:
        genes_node_full["New_Gene_ID"] = genes_node_full["Entrez_ID"].map(gtex_id_map)
        genes_node_full['Gencode_ID'] = genes_node_full['Gencode_ID'].combine_first(genes_node_full['New_Gene_ID'])
        genes_node_full.drop(["New_Gene_ID"], axis = 1, inplace = True)
        genes_node_full = genes_node_full[all_cols]

    # Removing duplicates
    cols = genes_node_full.columns
    flags = [i for i in cols if "Flag" in i]

    genes_node_full["pLI_Imputed_Flag"] = genes_node_full["pLI_Imputed_Flag"].astype(float)
    genes_node_full["Gene_Length_Imputed_Flag"] = genes_node_full["Gene_Length_Imputed_Flag"].astype(float)
    genes_node_full["flag_sum"] = genes_node_full[flags].sum(axis=1)

    # Remove duplicates based on Ensembl_ID, keeping the row with the lowest flag_sum (i.e., the most complete data)
    df_geneId_clean = genes_node_full[~genes_node_full["Gencode_ID"].isna()]
    df_geneId_na = genes_node_full[genes_node_full["Gencode_ID"].isna()]

    df_clean_dedup_ensemble = df_geneId_clean.sort_values(["flag_sum"]).drop_duplicates(subset=['Ensembl_ID'], keep='first')
    genes_node_dedup_ensemble = pd.concat([df_clean_dedup_ensemble, df_geneId_na], axis = 0, ignore_index = True)

    # Remove duplicates based on Gene_Symbol, keeping the row with the lowest flag_sum (i.e., the most complete data)
    genes_node_dedup_map_symbols = genes_node_dedup_ensemble.sort_values(["Ensembl_ID"]).drop_duplicates(subset=['Gene_Symbol'], keep='first')

    consolidation_map = {
            "C10orf11":"LRMDA",            # Previous fix
            "LOC105371814":"SUMO2P17",    # Consolidate structural/regulatory edges
            "LOC729683":"RP11-51F16.1",  # Consolidate structural/regulatory edges
            "PRSS45P":"PRSS45",         # Move  Pseudogene edges to node with real features
            "FAM60A":"SINHCAF"        # Move edges to node with real features

        }

    genes_node_dedup_map_symbols["Gene_Symbol"] = genes_node_dedup_map_symbols["Gene_Symbol"].replace(consolidation_map)

    # Write the final gene features and gene node data to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_features_output_path = output_dir / "gene_features_imputted.csv"
    genes_node_dedup_mapped_output_path = output_dir / "genes_node_dedup_mapped.csv"

    genes_node_full.to_csv(gene_features_output_path, index=False)
    genes_node_dedup_map_symbols.to_csv(genes_node_dedup_mapped_output_path, index=False)

    return gene_features_output_path, genes_node_dedup_mapped_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene features and imputation for missing data")
    parser.add_argument("--unique-gene-list", default="df_gene_collapsed.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--vg-edges", default="vg_edges.csv", help="Path to the variant-gene edge CSV")
    parser.add_argument("--variant-tissue-edges", default="variant_tissue_edges_final.csv", help="Path to the V_G_T edge CSV")
    parser.add_argument("--gene-features", default="gene_features_all.csv", help="Path to the gene features CSV")
    parser.add_argument("--vocab-size", default="75", help="Vocabulary size for the gene features")
    parser.add_argument("--output-dir", default="output/gene_lists", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = gene_features_imputation(
        unique_gene_list=Path(args.unique_gene_list),
        vg_edges=Path(args.vg_edges),
        variant_tissue_edges=Path(args.variant_tissue_edges),
        gene_features=Path(args.gene_features),
        output_dir=Path(args.output_dir),
        vocab_size=int(args.vocab_size)
    )
    print(f"Wrote the imputted gene features to {output_paths[0]} and gene deduplicated node data to {output_paths[1]}")


if __name__ == "__main__":
    main()
