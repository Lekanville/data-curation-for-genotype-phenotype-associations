import requests
import argparse
import mygene
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# Get entrez and ensemble IDs
def fetch_all_gene_ids_batch(symbol_list):
    mg = mygene.MyGeneInfo()
    
    print(f"Batch mapping {len(symbol_list)} identifiers...")
    
    # We use multiple scopes to handle both official symbols 
    # and the 9606.ENSP protein IDs you added
    results = mg.querymany(
        symbol_list, 
        scopes='symbol,ensembl.protein,entrezgene', 
        fields='entrezgene,ensembl.gene', 
        species='human', 
        as_dataframe=True
    )
    
    # Reset index so 'query' becomes a column
    df_results = results.reset_index()
    
    # 1. Clean up Ensembl IDs (handle cases with multiple matches)
    def clean_ensembl(val):
        if isinstance(val, list): return val[0]['gene']
        if isinstance(val, dict): return val.get('gene')
        return val

    # 2. Extract and format the data
    df_mapped = pd.DataFrame()
    df_mapped['Gene_Symbol'] = df_results['query']
    df_mapped['Entrez_ID'] = df_results['entrezgene']
    df_mapped['Ensembl_ID'] = df_results['ensembl'].apply(clean_ensembl)
    
    # 3. Handle duplicates (MyGene might return multiple rows for one query)
    df_mapped = df_mapped.drop_duplicates(subset=['Gene_Symbol'], keep='first')
    
    print(f"Mapping complete. Successfully found Ensembl IDs for {df_mapped['Ensembl_ID'].notna().sum()} nodes.")
    return df_mapped

# Get Genecode IDs from Ensemble IDS
def resolve_ensembl_ids(ensembl_base_ids):
    """
    Maps base Ensembl IDs to the versioned GENCODE IDs used in GTEx v8.
    """
    url = "https://gtexportal.org/api/v2/reference/gene"
    resolved_mapping = []
    
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
                entrez_id = str(entry['entrezGeneId'])
                gene_symbol = entry['geneSymbol']
                gencode_id = entry['gencodeId']
                
                resolved_mapping.append({'Entrez_ID':entrez_id, 'Gene_Symbol':gene_symbol, 'Gencode_ID':gencode_id})
                
        except Exception as e:
            print(f"❌ Error resolving chunk: {e}")
            
    return pd.DataFrame(resolved_mapping)


def build_gene_list(
    modifiers_and_vgt_gene_list_path: Path, 
    ppi_path: Path, 
    vg_edges_path: Path, 
    output_dir: Path
    ) -> Path:

    # Load the gene list and PPI data
    modifiers_and_VT_gene_list = pd.read_csv(modifiers_and_vgt_gene_list_path)
    df_ppi_final = pd.read_csv(ppi_path)
    df_gene_from_modifiers = pd.read_csv(vg_edges_path)

    # Get the unique set of all genes currently in GG PPI edges
    gg_sources = list(df_ppi_final["source"].unique())
    gg_targets = list(df_ppi_final["target"].unique())
    ppi_gene_set = set(gg_sources + gg_targets)

    # Identify which ones are NOT already in your master list
    existing_gene_set = set(modifiers_and_VT_gene_list['Gene_Symbol'])
    new_genes_to_add = list(ppi_gene_set - existing_gene_set)
    print(f"Adding {len(new_genes_to_add)} new genes to the node list.")

    # Create a temporary dataframe for the new genes
    # We initialize columns to NaN so your 'get_ids' code can fill them later
    df_new_nodes = pd.DataFrame({'Gene_Symbol': new_genes_to_add})

    # Combine with the existing list
    df_all_genes_updated = pd.concat([modifiers_and_VT_gene_list, df_new_nodes], ignore_index=True)

    # Safety check: Remove any accidental duplicates
    df_all_genes_updated = df_all_genes_updated.drop_duplicates(subset=['Gene_Symbol'])

    # Clean the gene symbols
    gene_id_list = df_all_genes_updated['Gene_Symbol'].unique()
    # cleaned_symbols = clean_gene_symbols(gene_id_list)

    # Use the unique set of genes from PPI and initial list
    df_mapped_final = fetch_all_gene_ids_batch(gene_id_list)

    # Merge back into your main node list
    df_gene_entrez_updated = pd.merge(df_all_genes_updated, df_mapped_final, on='Gene_Symbol', how='outer')

    # Handle cases where Entrez_ID might be missing in one of the merged dataframes
    df_gene_entrez_updated["Entrez_ID_y"] = df_gene_entrez_updated["Entrez_ID_y"].astype(float)
    df_gene_entrez_updated["Entrez_ID"] = df_gene_entrez_updated["Entrez_ID_x"].combine_first(df_gene_entrez_updated["Entrez_ID_y"])
    df_gene_entrez_updated.drop(["Entrez_ID_x", "Entrez_ID_y"], axis = 1, inplace = True)

    # Now we resolve the Ensembl IDs to Gencode IDs using the GTEx API
    ensembl_to_resolve = df_gene_entrez_updated['Ensembl_ID'].dropna().unique().tolist()
    print(f"Resolving versioned IDs for {len(ensembl_to_resolve)} genes via GTEx API...")
    ensemble_resolved_df = resolve_ensembl_ids(ensembl_to_resolve)

    # Merge this back to your master dataframe. This will fill in those 'Gencode_ID' NaNs in one go
    df_gene_genecode_updated = pd.merge(
        df_gene_entrez_updated, 
        ensemble_resolved_df, 
        on='Gene_Symbol', 
        how='outer',
        suffixes=('', '_new')
    )

    # Update the Gencode_ID column with the new versioned IDs
    df_gene_genecode_updated['Gencode_ID'] = df_gene_genecode_updated['Gencode_ID'].fillna(df_gene_genecode_updated['Gencode_ID_new'])
    df_gene_genecode_updated.drop(columns=['Gencode_ID_new'], inplace=True)

    # Handle cases where Ensembl_ID might be missing but Gencode_ID is present
    df_gene_genecode_updated["Ensembl_ID"] = df_gene_genecode_updated["Ensembl_ID"].fillna(
        df_gene_genecode_updated["Gencode_ID"].str.split(".").str[0]
    )

    # A temporary key to ensure we don't lose the ~1,800 genes without Ensembl IDs
    df_gene_genecode_updated['Mapping_Key'] = df_gene_genecode_updated['Ensembl_ID'].fillna(df_gene_genecode_updated['Gene_Symbol'])

    # Collapse based on this hybrid key
    df_gene_collapsed = df_gene_genecode_updated.groupby('Mapping_Key').agg({
        'Gene_Symbol': 'first',
        'Ensembl_ID': 'first',
        'Gencode_ID': 'first',
        'Entrez_ID': 'first'
    }).reset_index(drop=True)

    # Verify the count
    print(f"Total gene nodes preserved: {len(df_gene_collapsed)}")

    # Write the final outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_list_output_path = output_dir / "df_gene_collapsed.csv"
    df_gene_collapsed.to_csv(gene_list_output_path, index=False)

    return gene_list_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full gene list from vg (modifiers) and V_G_T outputs")
    parser.add_argument("--modifiers-and-vgt-gene-list", default="modifiers_and_VGT_gene_list.csv", help="Path to the modifiers_and_VGT_gene_list CSV")
    parser.add_argument("--ppi", default="df_ppi_final.csv", help="Path to the PPI CSV")
    parser.add_argument("--vg-edges", default="vg_edges.csv", help="Path to the variant-gene edge CSV")
    parser.add_argument("--output-dir", default="output/gene_sources_and_list", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_gene_list(
        modifiers_and_vgt_gene_list_path=Path(args.modifiers_and_vgt_gene_list),
        ppi_path=Path(args.ppi),
        vg_edges_path=Path(args.vg_edges),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote the gene list output to {output_paths}")


if __name__ == "__main__":
    main()
