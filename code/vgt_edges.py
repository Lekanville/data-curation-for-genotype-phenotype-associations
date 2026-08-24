import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests


def resolve_variant_ids_to_gtex(rsid_list: List[str]) -> pd.DataFrame:
    """Convert RSIDs to GTEx-compatible variant IDs using the GTEx API."""
    url = "https://gtexportal.org/api/v2/dataset/variant"
    variant_map = []
    chunk_size = 50

    for rsid in rsid_list:
        params = {"snpId": rsid, "datasetId": "gtex_v8"}
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get("data", [])
            for entry in data:
                variant_map.append({"rsid": entry.get("snpId"), "Variant_ID": entry.get("variantId")})
            time.sleep(0.1)
        except Exception as exc:
            print(f"❌ Error resolving variant {rsid}: {exc}")

    return pd.DataFrame(variant_map).drop_duplicates()


def get_variant_tissue_edges(gtex_mapped_ids: List[str], tissue_ids: List[str]) -> pd.DataFrame:
    """Query GTEx eQTL associations for the resolved variants across tissues."""
    url = "https://gtexportal.org/api/v2/association/singleTissueEqtl"
    eqtl_results = []

    print(f"Querying eQTLs for {len(gtex_mapped_ids)} variants across {len(tissue_ids)} tissues...")

    for variant_id in gtex_mapped_ids:
        params = {
            "variantId": variant_id,
            "tissueSiteDetailId": tissue_ids,
            "datasetId": "gtex_v8",
            "p_value_threshold": 0.05,
        }
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get("data", [])
            for entry in data:
                eqtl_results.append({
                    "Variant_ID": variant_id,
                    "Tissue_ID": entry.get("tissueSiteDetailId"),
                    "P_Value": entry.get("pValue"),
                    "NES": entry.get("nes"),
                    "Gene_ID": entry.get("gencodeId"),
                })
            time.sleep(0.05)
        except Exception as exc:
            print(f"❌ Error fetching eQTLs for {variant_id}: {exc}")

    return pd.DataFrame(eqtl_results)


def build_variant_tissue_edges(variant_node_path: Path, tissue_node_path: Path, output_dir: Path) -> Path:
    df_variants = pd.read_csv(variant_node_path)
    df_tissue_nodes = pd.read_csv(tissue_node_path)

    rsid_list = df_variants["rsid"].dropna().astype(str).tolist()
    variant_id_map = resolve_variant_ids_to_gtex(rsid_list)

    mapped_variants = variant_id_map["Variant_ID"].dropna().astype(str).tolist()
    df_variant_tissue_edges = get_variant_tissue_edges(mapped_variants, df_tissue_nodes["Tissue_ID"].dropna().astype(str).tolist())

    threshold = 1e-5
    df_eqtl_filtered = df_variant_tissue_edges[df_variant_tissue_edges["P_Value"] < threshold].copy()
    df_eqtl_filtered["Ensembl_ID"] = df_eqtl_filtered["Gene_ID"].astype(str).str.split(".").str[0]

    df_variant_tissue_edges_final = pd.merge(variant_id_map, df_eqtl_filtered, on="Variant_ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "variant_tissue_edges_final.csv"
    df_variant_tissue_edges_final.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V_G_T edges from variant and tissue node outputs")
    parser.add_argument("--variant-node", default="variant_node.csv", help="Path to the variant node CSV")
    parser.add_argument("--tissue-node", default="tissue_node.csv", help="Path to the tissue node CSV")
    parser.add_argument("--output-dir", default="output/vgt_edges", help="Directory for the generated edge CSV")
    args = parser.parse_args()

    output_path = build_variant_tissue_edges(
        Path(args.variant_node),
        Path(args.tissue_node),
        Path(args.output_dir),
    )
    print(f"The Variant-Gene-Tissue edges are saved to {output_path}")


if __name__ == "__main__":
    main()
