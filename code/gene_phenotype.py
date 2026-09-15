import io
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def build_gene_phenotype(
    unique_gene_node: Path, 
    variant_phenotype_nodes: Path, 
    variant_gene_edges: Path,
    include_clinical_phenotypes: bool,
    output_dir: Path
    ) -> Tuple[Path, Path]:

    # Load the unique gene node file, varaint-phenotype edges, and variant-gene edges
    genes_node_full = pd.read_csv(unique_gene_node)
    df_vp_edges = pd.read_csv(variant_phenotype_nodes)
    df_vg_master = pd.read_csv(variant_gene_edges)

    if include_clinical_phenotypes == True:
        # Get gene-phenotype data form OBO database ans extrace our genes of interest using Entrez IDs and gene symbols
        hpo_purl = "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
        try:
            # Use a generic User-Agent to prevent the server from blocking the request
            response = requests.get(hpo_purl, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            # Read the file. We must skip lines starting with '#' (metadata)
            hpo_raw = pd.read_csv(io.StringIO(response.text), sep='\t', comment='#', header=None, low_memory = False)
            
            # Assign the 2026 HPO column standard
            # Manually set the 6 column names seen in your successful load
            hpo_raw.columns = ['ncbi_gene_id', 'gene_symbol', 'hpo_id', 'hpo_name', 'frequency', 'disease_id']
            
            # Extract your elite symbols and IDs
            # We use both to ensure no gene is left behind
            genes_node_full["Entrez_ID"] = genes_node_full["Entrez_ID"].astype(str).str.split(".").str[0]
            current_entrez_ids = genes_node_full[genes_node_full["Entrez_ID"] != "nan"]["Entrez_ID"].tolist()

            # elite_ids = set(genes_node_full['Entrez_ID'].dropna().unique().astype(int))
            elite_symbols = set(genes_node_full['Gene_Symbol'].unique())
            
            # Filter using the existing data in memory
            df_gp_edges = hpo_raw[
                (hpo_raw['ncbi_gene_id'].isin(current_entrez_ids)) | 
                (hpo_raw['gene_symbol'].isin(elite_symbols))
            ].copy()
            
            # Final Edge List: Source (Gene) -> Target (HPO Phenotype)
            df_gp_master = df_gp_edges[['gene_symbol', 'hpo_id', 'hpo_name']].drop_duplicates()
            df_gp_master.columns = ['source', 'target', 'phenotype_label']
            
            print(f"Success! Created {len(df_gp_master)} Gene-Phenotype edges.")

        except Exception as e:
            print(f"Deep error check: {e}")
        
        # Now we will get the gene-phenotype edges using both the curated HPO data and the GWAS data. 
        # We will assign weights to each edge based on the source of the data.  

        # 1. Start with the 'Clinical Truth' (The 96 HPO Genes)
        # These get a high confidence weight of 1.0 because they are curated
        df_gp_clinical = df_gp_master.copy()
        df_gp_clinical['weight'] = 1.0

    # 2. Derive weights from your Variant-Phenotype (GWAS) data
    # We use -log10(p) to turn small p-values into large edge weights.
    # Keep the 1e-300 safeguard for exact zeros, but normalize within each phenotype
    # so we do not incorrectly divide by the max of only one trait.
    df_vp_edges = df_vp_edges.copy()
    df_vp_edges['p_safe'] = df_vp_edges['p'].replace(0, 1e-300)
    df_vp_edges['neg_log_p'] = -np.log10(df_vp_edges['p_safe'])

    max_neg_log_p_per_trait = df_vp_edges.groupby('trait')['neg_log_p'].transform('max')
    max_neg_log_p_per_trait = max_neg_log_p_per_trait.replace(0, 1e-300)
    df_vp_edges['stat_weight'] = 0.4 + 0.4 * (df_vp_edges['neg_log_p'] / max_neg_log_p_per_trait)

    # 3. Collapse Variant weights to Gene Symbols
    # Build the trait-to-HPO map directly from the variant-phenotype table so new phenotypes are handled automatically.
    trait_to_hpo = (
        df_vp_edges[['trait', 'target_phenotype']]
        .drop_duplicates()
        .dropna(subset=['target_phenotype'])
        .set_index('trait')['target_phenotype']
        .to_dict()
    )
    df_vp_edges['trait_name'] = df_vp_edges['trait'].fillna('Unknown')
    df_vp_edges['target'] = df_vp_edges['trait_name'].map(trait_to_hpo)
    df_vp_edges['target'] = df_vp_edges['target'].fillna(df_vp_edges.get('target_phenotype', pd.Series([np.nan] * len(df_vp_edges))))

    # Merge with the gene mapping table on rsid; gene targets are kept as source genes.
    df_vg_master_clean = df_vg_master.rename(columns={'source': 'rsid', 'target': 'gene_symbol'})
    df_gwas_gp = pd.merge(df_vp_edges, df_vg_master_clean, on='rsid', how='inner')

    # Take the maximum statistical weight if multiple SNPs hit the same gene for the same phenotype.
    df_gp_statistical = (
        df_gwas_gp.groupby(['target', 'gene_symbol'], as_index=False)
        .agg(weight=('stat_weight', 'max'))
    )
    df_gp_statistical['phenotype_label'] = df_gp_statistical['target'].map(
        {v: k for k, v in trait_to_hpo.items()}
    ).fillna('Unknown')
    df_gp_statistical = df_gp_statistical.rename(columns={'gene_symbol': 'source'})
    df_gp_statistical = df_gp_statistical[['source', 'target', 'phenotype_label', 'weight']]

    # 4. Final Integration
    # Clinical edges already have their own HPO target; statistical edges keep the trait-mapped HPO target.
    if include_clinical_phenotypes == True:
        df_final_gp = pd.concat([
            df_gp_clinical,
            df_gp_statistical[['source', 'target', 'phenotype_label', 'weight']]
        ])
    else:
        df_final_gp = df_gp_statistical[['source', 'target', 'phenotype_label', 'weight']]

    df_final_gp = (
        df_final_gp.sort_values(by=['source', 'target', 'weight'], ascending=[True, True, False])
        .drop_duplicates(subset=['source', 'target'])
        .reset_index(drop=True)
    )

    df_final_gp = df_final_gp[["source", "target", "phenotype_label", "weight"]].copy()

    # Get the new phenotypes
    new_phenotypes = df_final_gp[['target', 'phenotype_label']].drop_duplicates()
    new_phenotypes.columns = ['HPO_ID', 'Phenotype_Name']
    new_phenotypes.drop_duplicates(subset = "HPO_ID", keep = 'first', inplace = True)

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_phenotype_edges_dir = output_dir / "gene_phenotype_edges.csv"
    new_phenotypes_dir = output_dir / "new_phenotypes_from_gp_edges.csv"

    df_final_gp.to_csv(gene_phenotype_edges_dir, index=False)
    new_phenotypes.to_csv(new_phenotypes_dir, index=False)


    return gene_phenotype_edges_dir, new_phenotypes_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene features and imputation for missing data")
    parser.add_argument("--unique-gene-node", default="vp_edges_refined.csv", help="Path to the variant-phenotype edge CSV")
    parser.add_argument("--variant-phenotype-edges", default="vp_edges_refined.csv", help="Path to the variant-phenotype edges CSV")
    parser.add_argument("--variant-gene-edges", default="df_vg_master_clean.csv", help="Path to the variant-gene edges CSV")
    parser.add_argument("--include-clinical-phenotypes", default=False, help="Handle to decide theinclusion of clinically related genotype-phenotype")
    parser.add_argument("--output-dir", default="output/gene_phenotype", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_gene_phenotype(
        unique_gene_node=Path(args.unique_gene_node),
        variant_phenotype_nodes=Path(args.variant_phenotype_edges),
        variant_gene_edges=Path(args.variant_gene_edges),
        include_clinical_phenotypes = args.include_clinical_phenotypes,
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the gene-phenotype edges data to {output_paths[0]}")
    print(f"Wrote (new) phenotypes to {output_paths[1]}.")


if __name__ == "__main__":
    main()
