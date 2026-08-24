import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def  build_variant_gene_edges(
    unique_gene_node: Path,
    vg_edges: Path,
    variant_tissue_edges: Path,
    output_dir: Path
    ) -> Tuple[Path, Path]:
    """
    Build the variant-gene edges from the unique gene node, variant-gene edges, and variant-tissue edges.
    """
    # The gene nodes
    genes_node_full = pd.read_csv(unique_gene_node)

    # Structural v2g edges
    df_gene_from_modifiers = pd.read_csv(vg_edges)

    # Regulatory v2g edges
    df_gene_from_vgt = pd.read_csv(variant_tissue_edges)

    # Merge duplicates because of ID mismatches from symbol mapping
    # Its better to remap it here because of the [source , target] mapping
    # Check all genes dict notebook to understand the renaming

    # Standardize symbols in your edge tables
    consolidation_map = {
            "C10orf11":"LRMDA",            # Previous fix
            "LOC105371814":"SUMO2P17",    # Consolidate structural/regulatory edges
            "LOC729683":"RP11-51F16.1",  # Consolidate structural/regulatory edges
            "PRSS45P":"PRSS45",         # Move  Pseudogene edges to node with real features
            "FAM60A":"SINHCAF"
        }

    consolidation_for_ensemble = {
            "ENSG00000139146":"ENSG00000276371"  # "FAM60A":"SINHCAF"
        }

    # Apply to Variant-Gene master list
    df_gene_from_modifiers["Target_Gene_ID"] = df_gene_from_modifiers["Target_Gene_ID"].replace(consolidation_map)

    # Apply to Variant-Tissue-Gene list if necessary
    df_gene_from_vgt["Ensembl_ID"] = df_gene_from_vgt['Ensembl_ID'].replace(consolidation_for_ensemble)


    all_genes = set(genes_node_full['Gene_Symbol'])
    all_ensemble = set(genes_node_full[~genes_node_full["Ensembl_ID"].isna()]["Ensembl_ID"].to_list())

    #1. Define the master impact mapping and apply to dataframe
    impact_weights = {
        'HIGH': 1.0,
        'MODERATE': 0.8,
        'LOW': 0.4,
        'MODIFIER': 0.1
    }

    # We fillna with 0 for genes that have no structural impact flag
    df_gene_from_modifiers['Structural_Weight'] = df_gene_from_modifiers['Edge_Feature_Impact'].map(impact_weights).fillna(0.0)

    # 2. Rename for consistency with Regulatory Edges
    df_gene_from_modifiers = df_gene_from_modifiers.rename(columns={
        'Source_Variant_rsid': 'source',
        'Target_Gene_ID': 'target'
    })

    # 4. Filter to ensure target exists in our Master List
    # df_vg_structural = df_vg_structural[df_vg_structural['target'].isin(valid_genes)]

    # Check for "Orphan Edges" (Targets that don't exist in the gene node list)
    orphan_targets_structural = df_gene_from_modifiers[~df_gene_from_modifiers['target'].isin(all_genes)]
    print(f"Number of edges pointing to missing nodes in structural edges: {len(orphan_targets_structural)}")

    orphan_targets_regulatory = df_gene_from_vgt[~df_gene_from_vgt['Ensembl_ID'].isin(all_ensemble)]
    print(f"Number of edges pointing to missing nodes in regulatory edges: {len(orphan_targets_regulatory)}")

    # Map Ensembl IDs to Gene Symbols for the regulatory edges
    ensemble_to_symbol = genes_node_full[~genes_node_full["Ensembl_ID"].isna()][["Gene_Symbol", "Ensembl_ID"]].set_index("Ensembl_ID").to_dict()
    df_gene_from_vgt["Gene_Symbol"] = df_gene_from_vgt["Ensembl_ID"].map(ensemble_to_symbol.get('Gene_Symbol'))

    # 3. PREPARE STRUCTURAL TABLE
    df_vg_structural = df_gene_from_modifiers[['source', 'target', 'Structural_Weight', 'Edge_Feature_Impact']]

    # 1. CLEAN REGULATORY (VGT) TABLE - Rename for consistency
    df_vg_regulatory = df_gene_from_vgt.rename(columns={'rsid': 'source','Gene_Symbol': 'target'})

    # Filter: Only keep edges where the target gene is in master gene list
    df_vg_regulatory = df_vg_regulatory[df_vg_regulatory['target'].isin(all_genes)]

    # 2. CALCULATE REGULATORY WEIGHT
    # Standard formula: Normalize |NES| * -log10(P-value) to reflect biological strength
    df_vg_regulatory['Regulatory_Weight'] = np.abs(df_vg_regulatory['NES']) * (-np.log10(df_vg_regulatory['P_Value']))
    df_vg_regulatory = df_vg_regulatory[['source', 'target', 'Regulatory_Weight', 'Tissue_ID']]


    # 4. THE MASTER MERGE (Outer Join)
    # This combines both tables. If a pair exists in both, it joins them in one row.
    df_vg_master = pd.merge(
        df_vg_structural, 
        df_vg_regulatory, 
        on=['source', 'target'], 
        how='outer'
    )

    # 5. HANDLE MISSING VALUES
    # If an edge is structural-only, Regulatory_Weight is 0.0. If regulatory-only, Structural_Weight is 0.0.
    df_vg_master['Structural_Weight'] = df_vg_master['Structural_Weight'].fillna(0.0)
    df_vg_master['Regulatory_Weight'] = df_vg_master['Regulatory_Weight'].fillna(0.0)
    df_vg_master['Tissue_ID'] = df_vg_master['Tissue_ID'].fillna('Global')

    df_vg_master['Regulatory_Weight_Log'] = np.log1p(df_vg_master['Regulatory_Weight'])

    # Consolidation with a 'Safety Max' for Regulatory Weights
    df_vg_master_clean = df_vg_master.groupby(['source', 'target']).agg({
        'Structural_Weight': 'max',   # Correct: Keep the binary-ish structural impact
        'Regulatory_Weight': 'max',   # Correct: Use 'max' to avoid 12.45 vs 0.9 scaling issues
        'Regulatory_Weight_Log': 'max',
        'Edge_Feature_Impact': 'first' 
    }).reset_index()


    scaler = MinMaxScaler()
    df_vg_master_clean['Regulatory_Weight_Norm'] = scaler.fit_transform(df_vg_master_clean[['Regulatory_Weight_Log']])
    # df_vg_master_clean['Regulatory_Weight_Norm'] = scaler.fit_transform(df_vg_master_clean[['Regulatory_Weight']])
    # df_vg_master_clean['Unified_Weight'] = (df_vg_master_clean['Structural_Weight'] + df_vg_master_clean['Regulatory_Weight_Norm']) / 2.0
    df_vg_master_clean['Unified_Weight'] = df_vg_master_clean[['Structural_Weight', 'Regulatory_Weight_Norm']].max(axis=1)

    print(f"The unique variant-gene edges after merging structural and regulatory edges: \n \n {df_vg_master_clean.head(5)} ")
    print(f"The final number of unique variant-gene edges: {len(df_vg_master_clean)}")

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    vg_edges_all = output_dir / "df_vg_master_all.csv"
    vg_edges_cleaned = output_dir / "df_vg_master_clean.csv"

    df_vg_master.to_csv(vg_edges_all, index=False)
    df_vg_master_clean.to_csv(vg_edges_cleaned, index=False)

    return vg_edges_all, vg_edges_cleaned



def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene features and imputation for missing data")
    parser.add_argument("--unique-gene-node", default="genes_node_dedup_mapped.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--vg-edges", default="vg_edges.csv", help="Path to the variant-gene edge CSV")
    parser.add_argument("--variant-tissue-edges", default="variant_tissue_edges_final.csv", help="Path to the V_G_T edge CSV")
    parser.add_argument("--output-dir", default="output/variant_gene_edges", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_variant_gene_edges(
        unique_gene_node=Path(args.unique_gene_node),
        vg_edges=Path(args.vg_edges),
        variant_tissue_edges=Path(args.variant_tissue_edges),
        output_dir=Path(args.output_dir),
    )

    print(f"The full variant-gene edges data is saved to {output_paths[0]} and the cleaned deduplicated variant-gene edges data to {output_paths[1]}")


if __name__ == "__main__":
    main()
