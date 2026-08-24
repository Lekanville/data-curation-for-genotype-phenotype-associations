import time
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def get_tissue_expression_edges(gencode_ids, tissue_ids):
    """
    Fetches median expression (TPM) from GTEx for a list of genes.
    Returns a DataFrame: Index = Gene_ID, Columns = Tissue_IDs
    """
    url = "https://gtexportal.org/api/v2/expression/medianGeneExpression"
    all_results = []
    
    # GTEx API usually limits to ~50-100 IDs per request
    chunk_size = 50
    for i in range(0, len(gencode_ids), chunk_size):
        chunk = gencode_ids[i:i + chunk_size]
        
        params = {
            'gencodeId': chunk,
            'tissueSiteDetailId': tissue_ids,
            'datasetId': 'gtex_v8'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # The API returns a list of dictionaries
            if data.get('data'):
                all_results.extend(data.get('data'))
            
            # Be kind to the API
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"❌ Error fetching chunk starting at {i}: {e}")
            
    if not all_results:
        return pd.DataFrame()

    # Convert to DataFrame
    df_raw = pd.DataFrame(all_results)
    
    df_pivot = df_raw.pivot(
        index='gencodeId', 
        columns='tissueSiteDetailId', 
        values='median'
    )
    
    # return df_raw, df_pivot
    return df_pivot

def finalize_gene_tissue_edges(df_pivot):
    """
    Takes the wide TPM matrix and turns it into a long edge list with Log weights.
    """
    # 1. Melt to Long format: [Gene, Tissue, TPM]
    edge_list = df_pivot.reset_index().melt(
        id_vars='gencodeId', 
        var_name='Tissue_ID', 
        value_name='TPM'
    )
    
    # 2. Rename columns for graph consistency
    edge_list.columns = ['Gene_ID', 'Tissue_ID', 'TPM']
    
    # 3. Apply Log10(TPM + 1)
    edge_list['Weight'] = np.log10(edge_list['TPM'] + 1)
    scaler = MinMaxScaler()
    edge_list['Weight_Norm'] = scaler.fit_transform(edge_list[['Weight']])
    
    # 4. Pruning: Remove zero or trace expression to keep the graph clean
    # A threshold of 0.1 Log10(TPM+1) is roughly 0.25 TPM
    edge_list = edge_list[edge_list['Weight'] > 0.1].copy()
    
    return edge_list


def build_gene_tissue_edges(
    unique_gene_node: Path,
    tissue_node_path: Path,
    output_dir: Path
    ) -> Tuple[Path, Path]:

    """
    Build the gene-tissues edges from the unique gene node, and tissue nodes.
    """
    # The gene nodes - The 10 core tissues
    genes_node_full = pd.read_csv(unique_gene_node)

    # The tissue nodes
    df_tissue_nodes = pd.read_csv(tissue_node_path)
    tissue_ids = list(df_tissue_nodes["Tissue_ID"])

    gtex_versioned_ids = genes_node_full[~genes_node_full['Gencode_ID'].isna()]['Gencode_ID'].tolist()

    df_gtex_matrix = get_tissue_expression_edges(gtex_versioned_ids, tissue_ids)
    df_gene_tissue_edges = finalize_gene_tissue_edges(df_gtex_matrix)

    found_genecodes = genes_node_full[~genes_node_full['Gencode_ID'].isna()][["Gene_Symbol", "Gencode_ID"]].copy()

    df_gene_tissue_edges_final = pd.merge(
        df_gene_tissue_edges, found_genecodes, left_on = "Gene_ID", right_on = "Gencode_ID", how = "left"
        )[["Gene_Symbol", "TPM", "Weight", 'Weight_Norm', "Tissue_ID", "Gencode_ID"]]

    print(f"The gene-tissue edges after merging structural and regulatory edges: \n \n {df_gene_tissue_edges_final.head(5)} ")

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_tissue_edges = output_dir / "gene_tissue_edges_final.csv"
    df_gene_tissue_edges_final.to_csv(gene_tissue_edges, index=False)

    return gene_tissue_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene features and imputation for missing data")
    parser.add_argument("--unique-gene-node", default="genes_node_dedup_mapped.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--tissue-node", default="tissue_node.csv", help="Path to the tissue node CSV")
    parser.add_argument("--output-dir", default="output/gene_lists", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_gene_tissue_edges(
        unique_gene_node=Path(args.unique_gene_node),
        tissue_node_path=Path(args.tissue_node),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the full variant-gene edges data to {output_paths}")


if __name__ == "__main__":
    main()
