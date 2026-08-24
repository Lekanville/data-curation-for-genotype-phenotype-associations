import time
import requests
import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def get_pathways_for_genes(entrez_ids):
    """
    Maps a list of Entrez IDs to Reactome Pathways.
    Returns a dataframe of Pathway IDs and their associated Genes.
    """
    # Reactome Mapping Service
    url = "https://reactome.org/AnalysisService/identifiers/projection/?interactors=false"
    headers = {"Content-Type": "text/plain", "Accept": "application/json"}
    
    # Send Entrez IDs as a newline-separated string
    data = "\n".join(map(str, entrez_ids))
    
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        pathway_list = []
        # Extract pathways from the 'pathways' summary
        for p in result.get('pathways', []):
            pathway_list.append({
                'Pathway_ID': p['stId'],
                'Pathway_Name': p['name'],
                'Gene_Count_In_Pathway': p['entities']['found'],
                'P_Value (FDR)': p['entities']['fdr'] # FDR for enrichment
            })

        token = result['summary']['token']
        return token, pd.DataFrame(pathway_list)
    except Exception as e:
        print(f"❌ Error fetching pathways: {e}")
        return pd.DataFrame()

def get_gene_pathway_edges(token, pathway_ids):
    """
    Uses the POST /token/{token}/found/all endpoint identified in your screenshot.
    Returns a dataframe of Gene -> Pathway edges.
    """
    url = f"https://reactome.org/AnalysisService/token/{token}/found/all"
    headers = {"Content-Type": "text/plain", "Accept": "application/json"}
    
    # We send the pathway IDs as a newline-separated string in the body
    data = "\n".join(pathway_ids)
    
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        mapping_data = response.json()
        
        edge_rows = []
        # The response is a list of pathway objects
        for pathway_obj in mapping_data:
            p_id = pathway_obj['pathway']
            # 'entities' contains the list of our matched genes
            for entity in pathway_obj.get('entities', []):
                # We want the original ID we submitted (Entrez ID)
                gene_id = entity.get('id')
                edge_rows.append({
                    'Gene_ID': str(gene_id),
                    'Pathway_ID': p_id
                })
        
        return pd.DataFrame(edge_rows).drop_duplicates()
    except Exception as e:
        print(f"❌ Error fetching edges: {e}")
        return pd.DataFrame()

def get_pathway_hierarchy(pathway_id):
    """
    Fetches the full hierarchy for a Reactome ID and identifies the Top-Level category.
    """
    url = f"https://reactome.org/ContentService/data/event/{pathway_id}/ancestors"
    
    try:
        res = requests.get(url)
        res.raise_for_status()
        paths = res.json()
        
        # 'paths' is a list of lists. Each sub-list is a path to the root.
        # The Top-Level Pathway is the START of the ancestral path.
        top_levels = set()
        for path in paths: # Iterate through all possible ancestral paths
            for node in path: # Look at every node in that path
                if node.get('schemaClass') == 'TopLevelPathway':
                    top_levels.add(node.get('displayName'))
                
        return list(top_levels)
    except Exception as e:
        # If it returns a 404, it might be a very new or very specific event
        return []


def build_gene_pathways(
    unique_gene_node: Path,
    output_dir: Path
    ) -> Tuple[Path, Path]:

    # Load the unique gene node file and extract Entrez IDs
    genes_node_full = pd.read_csv(unique_gene_node)
    genes_node_full["Entrez_ID"] = genes_node_full["Entrez_ID"].astype(str).str.split(".").str[0]
    current_entrez_ids = genes_node_full[genes_node_full["Entrez_ID"] != "nan"]["Entrez_ID"].tolist()

    # Get pathways for the current Entrez IDs
    token, df_pathway_nodes_raw = get_pathways_for_genes(current_entrez_ids)

    # Get Gene-Pathway edges using the token and pathway IDs
    pathway_id_list = df_pathway_nodes_raw['Pathway_ID'].tolist()
    df_gene_pathway_edges = get_gene_pathway_edges(token, pathway_id_list)
    print(f"Found {len(df_gene_pathway_edges)} Gene-Pathway connections.")

    # Map Entrez IDs back to Gene Symbols for the final edge list
    found_entrez = genes_node_full[genes_node_full["Entrez_ID"] != "nan"][["Gene_Symbol", "Entrez_ID"]]
    df_gene_pathway_edges_final = pd.merge(
            df_gene_pathway_edges, found_entrez, left_on = "Gene_ID", right_on = "Entrez_ID", how = "left"
        )[["Gene_Symbol", "Entrez_ID",	"Pathway_ID"]]


    # Add the hierarchy information to the pathway nodes
    df_pathway_nodes_raw['Hierarchy'] = df_pathway_nodes_raw['Pathway_ID'].apply(get_pathway_hierarchy)

    # Standard threshold in bioinformatics is 0.05
    fdr_threshold = 0.05

    # Create the binary flag
    # Note: We use the FDR (Adjusted P-Value) for this to be statistically rigorous
    df_pathway_nodes_raw['Is_Enriched'] = (df_pathway_nodes_raw['P_Value (FDR)'] < fdr_threshold).astype(int)

    # Explode the hierarchy list into separate rows to get unique categories
    all_categories = set([cat for sublist in df_pathway_nodes_raw['Hierarchy'] for cat in sublist])

    # Build the Feature Matrix
    for cat in all_categories:
        df_pathway_nodes_raw[f'Is_{cat.replace(" ", "_")}'] = df_pathway_nodes_raw['Hierarchy'].apply(lambda x: 1 if cat in x else 0)

    # Set pathway ID as index
    df_pathway_nodes_raw.set_index('Pathway_ID', inplace=True)

    # Scale Gene_Count_In_Pathway
    scaler = MinMaxScaler()
    df_pathway_nodes_raw['Gene_Count_Normalized'] = scaler.fit_transform(df_pathway_nodes_raw[['Gene_Count_In_Pathway']])

    # Clean up the final Node Dataframe
    df_pathway_nodes = df_pathway_nodes_raw.drop(columns=['Hierarchy'])

    print(f"✅ Pathway Node Matrix (X_Path) initialized with {len(all_categories)} hierarchy features.")

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_pathway_edges_dir = output_dir / "gene_pathway_edges.csv"
    df_pathway_nodes_raw_dir = output_dir / "pathway_nodes_raw.csv"
    df_pathway_nodes_dir = output_dir / "pathway_nodes.csv"
    

    df_gene_pathway_edges_final.to_csv(gene_pathway_edges_dir, index=False)
    df_pathway_nodes_raw.to_csv(df_pathway_nodes_raw_dir, index=True)
    df_pathway_nodes.to_csv(df_pathway_nodes_dir, index=True)

    return gene_pathway_edges_dir, df_pathway_nodes_dir




def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene-pathway data")
    parser.add_argument("--unique-gene-node", default="genes_node_dedup_mapped.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--output-dir", default="output/gene_pathway", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_gene_pathways(
        unique_gene_node=Path(args.unique_gene_node),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the gene_pathway edges data to {output_paths[0]} and the pathway nodes data to {output_paths[1]}.")


if __name__ == "__main__":
    main()
