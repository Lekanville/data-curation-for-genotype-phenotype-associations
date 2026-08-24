import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


STRING_PHYSICAL_LINKS_URL = "https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz" # to get the high-confidence physical interaction edges from STRING
STRING_ALIAS_URL = "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz" # to get the mapping of STRING IDs to gene symbols
GTEX_GENE_URL = "https://gtexportal.org/api/v2/reference/gene" # to get the mapping of base Ensembl IDs to GENCODE IDs used in GTEx
STRING_ID_MAP_URL = "https://string-db.org/api/tsv/get_string_ids?" # to get the mapping of gene symbols to STRING IDs


def download_if_missing(url: str, target_path: Path) -> Path:
    """Download a URL to disk only if the target file does not already exist."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"Using cached file: {target_path}")
        return target_path

    print(f"Downloading {url} -> {target_path}")
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def resolve_ensembl_ids(ensembl_base_ids: List[str]) -> pd.DataFrame:
    """Map base Ensembl IDs to the versioned GENCODE IDs used in GTEx."""
    resolved_mapping = []
    chunk_size = 50

    for start in range(0, len(ensembl_base_ids), chunk_size):
        chunk = ensembl_base_ids[start:start + chunk_size]
        params = {"geneId": chunk, "datasetId": "gtex_v8"}
        try:
            response = requests.get(GTEX_GENE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get("data", [])
            for entry in data:
                resolved_mapping.append({
                    "Entrez_ID": str(entry.get("entrezGeneId", "")),
                    "Gene_Symbol": entry.get("geneSymbol"),
                    "Gencode_ID": entry.get("gencodeId"),
                })
            time.sleep(0.1)
        except Exception as exc:
            print(f"❌ Error resolving chunk: {exc}")

    return pd.DataFrame(resolved_mapping)


def gene_symbols_to_string_id(df_with_gene_symbols: pd.DataFrame) -> Tuple[dict, list]:
    """Map gene symbols from the gene list to STRING IDs."""
    gene_symbols_to_query = df_with_gene_symbols["Gene_Symbol"].dropna().astype(str).unique().tolist()
    string_id_to_symbol_map = {}
    batch_size = 200

    print("1. Mapping Gene Symbols to STRING IDs...")

    for start in range(0, len(gene_symbols_to_query), batch_size):
        batch = gene_symbols_to_query[start:start + batch_size]
        identifiers = "%0d".join(batch)
        params = {"identifiers": identifiers, "species": 9606, "caller_identity": "our_project"}
        try:
            response = requests.get(STRING_ID_MAP_URL, params=params, timeout=60)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                header = lines[0].split("\t")
                pref_name_idx = header.index("preferredName")
                string_id_idx = header.index("stringId")
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) > max(pref_name_idx, string_id_idx):
                        symbol_used = parts[pref_name_idx]
                        string_id = parts[string_id_idx]
                        string_id_to_symbol_map[string_id] = symbol_used
            time.sleep(0.5)
        except requests.exceptions.RequestException as exc:
            print(f"❌ Error during STRING MAP query: {exc}")
            time.sleep(5)

    official_string_ids = list(string_id_to_symbol_map.keys())
    print(f"Mapped {len(official_string_ids)} Symbols to STRING IDs.")
    return string_id_to_symbol_map, official_string_ids


def load_ppi_edges(ppi_file_path: Path, min_score: int = 700) -> pd.DataFrame:
    """Load the high-confidence STRING physical interaction edges from a local cached file."""
    df_ppi = pd.read_csv(ppi_file_path, sep=" ", compression="gzip")
    return df_ppi[df_ppi["combined_score"] >= min_score].copy()


def load_string_aliases(alias_file_path: Path) -> pd.DataFrame:
    """Load STRING aliases and keep the gene-symbol-like sources."""
    df_alias = pd.read_csv(alias_file_path, sep="\t", compression="gzip", skiprows=1, names=["string_id", "alias", "source"])
    return df_alias[df_alias["source"].isin(["Gene_Symbol", "HGNC_symbol", "Ensembl_HGNC_symbol"])]


def build_gene_gene_edges(
        vg_edges_path: Path,
        variant_tissue_edges_path: Path,
        output_dir: Path,
        cache_dir: Path,
    ) -> Tuple[Path, Path]:
    df_gene_from_modifiers = pd.read_csv(vg_edges_path)
    unique_from_modifiers = df_gene_from_modifiers.groupby(["Target_Gene_ID"]).first().reset_index().drop(["Source_Variant_rsid", "Edge_Feature_Impact"], axis=1)

    df_gene_from_vgt = pd.read_csv(variant_tissue_edges_path)
    unique_from_vgt = df_gene_from_vgt.groupby(["Gene_ID"]).first().reset_index().drop(["rsid", "Variant_ID", "Tissue_ID", "P_Value", "NES"], axis=1)
    genes_from_vgt = unique_from_vgt["Ensembl_ID"].dropna().astype(str).tolist()

    gtex_id_map = resolve_ensembl_ids(genes_from_vgt)

    modifiers_and_vgt_gene_list = pd.merge(unique_from_modifiers, gtex_id_map, left_on="Target_Gene_ID", right_on="Gene_Symbol", how="outer")
    modifiers_and_vgt_gene_list["Gene_Symbol"] = modifiers_and_vgt_gene_list["Gene_Symbol"].combine_first(modifiers_and_vgt_gene_list["Target_Gene_ID"])
    modifiers_and_vgt_gene_list = modifiers_and_vgt_gene_list.drop("Target_Gene_ID", axis=1)

    string_id_to_symbol_map, _ = gene_symbols_to_string_id(modifiers_and_vgt_gene_list)
    df_id_map = pd.DataFrame(list(string_id_to_symbol_map.items()), columns=['string_id', 'Gene_Symbol'])
    df_id_map['string_id'] = df_id_map['string_id'].apply(lambda x: x if x.startswith('9606.') else f"9606.{x}")


    ppi_file_path = download_if_missing(STRING_PHYSICAL_LINKS_URL, cache_dir / "9606.protein.physical.links.v12.0.txt.gz")
    alias_file_path = download_if_missing(STRING_ALIAS_URL, cache_dir / "9606.protein.aliases.v12.0.txt.gz")

    loaded_ppis = load_ppi_edges(ppi_file_path)
    df_symbols = load_string_aliases(alias_file_path)

    # 1. Map Protein 1 (Must be in your 463 genes)
    df_expanded = pd.merge(loaded_ppis, df_id_map, left_on='protein1', right_on='string_id', how='inner')

    # 2. Map Protein 2 (Allow it to be ANY protein in STRING)
    # We use 'left' join so we keep the interaction even if Gene 2 isn't in your 463 list
    df_expanded = pd.merge(df_expanded, df_id_map, left_on='protein2', right_on='string_id', how='left')

    # 3. Clean up
    df_expanded['Gene_2'] = df_expanded['Gene_Symbol_y'].fillna(df_expanded['protein2'])
    df_ppi = df_expanded[['Gene_Symbol_x', 'Gene_2', 'combined_score']]
    df_ppi.columns = ["source", "target", "combined_score"]
    print(f"Expanded PPI edges: {len(df_ppi)}")

    strings_to_get = df_ppi[df_ppi["target"].astype(str).str.startswith("9606.")]["target"].unique()
    mapping_dict = df_symbols[df_symbols["string_id"].isin(strings_to_get)].set_index("string_id")["alias"].to_dict()

    df_ppi["target_from_strings"] = df_ppi["target"].astype(str).map(mapping_dict).fillna(df_ppi["target"])
    df_ppi_final = df_ppi.drop("target", axis=1)
    df_ppi_final.rename(columns={"target_from_strings": "target"}, inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    edges_output_path = output_dir / "df_ppi_final.csv"
    gene_list_output_path = output_dir / "modifiers_and_VGT_gene_list.csv"
    df_ppi_final.to_csv(edges_output_path, index=False)
    modifiers_and_vgt_gene_list.to_csv(gene_list_output_path, index=False)

    return edges_output_path, gene_list_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gene-gene interaction edges from vg and V_G_T outputs")
    parser.add_argument("--vg-edges", default="vg_edges.csv", help="Path to the variant-gene edge CSV")
    parser.add_argument("--variant-tissue-edges", default="variant_tissue_edges_final.csv", help="Path to the V_G_T edge CSV")
    parser.add_argument("--output-dir", default="output/gene_gene", help="Directory for the gene-gene outputs")
    parser.add_argument("--cache-dir", default="data/string_cache", help="Directory for cached STRING download files")
    args = parser.parse_args()

    output_paths = build_gene_gene_edges(
        vg_edges_path=Path(args.vg_edges),
        variant_tissue_edges_path=Path(args.variant_tissue_edges),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
    )
    print(f"Wrote gene-gene edge outputs to {output_paths[0]} and {output_paths[1]}")


if __name__ == "__main__":
    main()
