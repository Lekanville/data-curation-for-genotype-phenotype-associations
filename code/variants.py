import os
import re
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ANCESTRY_LIST = [
    "European",
    "South Asian",
    "East Asian",
    "Hispanic or Latin American",
    "African unspecified",
    "Greater Middle East"
]

EXCLUSION_PHRASES = [
    "medication for",
    "self-reported",
    "do not have",
    "ever had diabetes",
    "gestational diabetes",
    "adjusted",
]

CONSEQUENCE_VOCABULARY = [
    # High/Moderate Impact
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "start_lost",
    "stop_lost",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "splice_region_variant",

    # Low/Modifier Impact
    "synonymous_variant",
    "intron_variant",
    "intergenic_variant",
    "intergenic_region",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
]

VARIANT_TYPE_VOCABULARY = [
    "snv", # Single Nucleotide Variant
    "del", # Explicit Deletion (Common dbSNP type)
    "ins", # Explicit Insertion (Common dbSNP type)
    "mnp",  # Multiple Nucleotide Polymorphism (Often used instead of 'mnv')
    "complex", # Used for variations involving multiple changes
    "other",
    "unknown", # For missing variant types
]

ANCESTRY_MAP = {
    'European': 'ANC_EUR',
    'East Asian': 'ANC_EAS',
    'South Asian': 'ANC_SAS',
    'Hispanic or Latin American': 'ANC_AMR',
    'African': 'ANC_SAF',
    # 'Southeast Asian': 'ANC_SEAS',
    'Oceanian': 'ANC_OCE',
    'Greater Middle East': 'ANC_GME'
}



def load_bearer_token() -> str:
    load_dotenv()
    token = os.getenv("BEARER_TOKEN")
    if not token:
        raise ValueError("BEARER_TOKEN is not set in the environment.")
    return token


def request_json(url: str, headers: Dict[str, str], method: str = "GET", payload: Optional[Dict] = None) -> Dict:
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    else:
        response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def select_studies(data_all_gwas: Dict, trait_pattern: str, exclude_phrases: Optional[List[str]] = None) -> pd.DataFrame:
    """Select studies matching a trait regex and apply exclusion phrases.

    Parameters:
        data_all_gwas: dict of studies from OpenGWAS
        trait_pattern: regex string to match the trait (case-insensitive)
        exclude_phrases: optional list of phrases to exclude

    Returns:
        DataFrame of prioritized studies (one per ancestry in ANCESTRY_LIST)
    """
    exclude_phrases = exclude_phrases or EXCLUSION_PHRASES
    all_matched = []
    pattern = re.compile(trait_pattern, re.IGNORECASE)

    for study_id, study_info in data_all_gwas.items():
        trait = str(study_info.get("trait", ""))
        if pattern.search(trait):
            study_info = dict(study_info)
            study_info["id"] = study_id
            all_matched.append(study_info)

    all_matched_df = pd.DataFrame(all_matched)

    def is_valid_trait(trait_text: str) -> bool:
        trait_lower = trait_text.lower()
        return not any(phrase in trait_lower for phrase in exclude_phrases)

    prioritized_rows = []
    if len(all_matched) > 0:
        for ancestry in ANCESTRY_LIST:
            studies_for_ancestry = all_matched_df[all_matched_df.get("population") == ancestry]
            if studies_for_ancestry.empty:
                continue

            filtered = studies_for_ancestry[studies_for_ancestry["trait"].fillna("").apply(is_valid_trait)].copy()
            if filtered.empty:
                continue

            filtered["ncase"] = pd.to_numeric(filtered["ncase"], errors="coerce").fillna(0)
            filtered["year"] = pd.to_numeric(filtered["year"], errors="coerce").fillna(1900)

            top = filtered.sort_values(by=["ncase", "year"], ascending=[False, False]).head(1)
            prioritized_rows.append(top)
    else:
        return pd.DataFrame()

    if prioritized_rows:
        return pd.concat(prioritized_rows, ignore_index=True)
    return pd.DataFrame()


def fetch_selected_gwasinfo(ids: List[str], headers: Dict[str, str]) -> pd.DataFrame:
    gwasinfo_url = "https://api.opengwas.io/api/gwasinfo"
    rows = []
    for study_id in ids:
        logging.info("Fetching GWAS info for %s", study_id)
        payload = {"id": study_id}
        data = request_json(gwasinfo_url, headers, method="POST", payload=payload)
        if data:
            rows.extend(data if isinstance(data, list) else [data])
    return pd.DataFrame(rows)


def fetch_tophits(ids: List[str], headers: Dict[str, str], pop: str = "EUR") -> pd.DataFrame:
    tophits_url = "https://api.opengwas.io/api/tophits"
    rows = []
    for study_id in ids:
        logging.info("Fetching top hits for %s", study_id)
        payload = {
            "id": study_id,
            "pval": 5e-8,
            "preclumped": 1,
            "clump": 1,
            "r2": 0.001,
            "kb": 5000,
            "pop": pop,
        }
        data = request_json(tophits_url, headers, method="POST", payload=payload)
        if data:
            rows.extend(data if isinstance(data, list) else [data])
    return pd.DataFrame(rows)


def load_african_catalogue(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        logging.info("Loading African GWAS catalogue from %s", path)
        return pd.read_csv(path)
    logging.info("African GWAS catalogue not found at %s, skipping.", path)
    return None


# def load_catalogues(paths: Optional[List[str]]) -> Optional[pd.DataFrame]:
def load_catalogues(paths: str) -> Optional[pd.DataFrame]:
    """Load zero or more CSV catalogue files and concatenate them.

    Returns None if no valid files provided.
    """
   
    if not paths:
        return None
    
    logging.info("Adding the included datasets")
    paths = paths.split("::")
    dfs = []
    
    for p in paths:
        pth = Path(p)
        if pth.exists():
            logging.info("Loading catalogue %s", p)
            try:
                dfs.append(pd.read_csv(pth))
            except Exception as e:
                logging.warning("Failed to read %s: %s", p, e)
        else:
            logging.warning("Catalogue file not found: %s", p)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True, axis=0)


def annotate_variants(opengwas_var: pd.DataFrame) -> pd.DataFrame:
    opengwas_var = opengwas_var.copy()
    opengwas_var["Variant_ID"] = ""
    opengwas_var["Inferred_type"] = "unknown"
    opengwas_var["Unique_SnpEff_Consequences"] = ""
    opengwas_var["consequence_multi_hot_vector"] = ""
    opengwas_var["Gene_Annotations"] = ""
    opengwas_var["CADD_Score"] = ""

    # Force Pandas to treat these columns as generic objects, NOT strings/floats
    opengwas_var["Unique_SnpEff_Consequences"] = opengwas_var["Unique_SnpEff_Consequences"].astype(object)
    opengwas_var["consequence_multi_hot_vector"] = opengwas_var["consequence_multi_hot_vector"].astype(object)
    opengwas_var["Gene_Annotations"] = opengwas_var["Gene_Annotations"].astype(object)

    for i, row in opengwas_var.iterrows():
        rsid = row.get("rsid")
        if not isinstance(rsid, str) or not rsid:
            continue

        mv_url = f"https://myvariant.info/v1/variant?id={rsid}&fields=all"
        try:
            vep_data = request_json(mv_url, {}, method="GET")
        except requests.exceptions.RequestException as err:
            logging.warning("Error querying MyVariant.info for %s: %s", rsid, err)
            continue

        if isinstance(vep_data, dict):
            vep_data = [vep_data]

        gwas_ea = row.get("ea")
        gwas_nea = row.get("nea")
        variant_record = None

        for record in vep_data:
            vcf_ref = record.get("vcf", {}).get("ref")
            vcf_alt = record.get("vcf", {}).get("alt")
            if row.get("source") == "Open GWAS" and gwas_ea == vcf_alt and gwas_nea == vcf_ref:
                variant_record = record
                break
            if row.get("source") == "GWAS Catalogue" and gwas_ea == vcf_alt:
                opengwas_var.at[i, "nea"] = vcf_ref
                variant_record = record
                break

        if not variant_record:
            logging.warning("No allele match found for rsid %s", rsid)
            continue

        variant_type_info = variant_record.get("dbsnp", {}).get("vartype")
        snpeff_ann = variant_record.get("snpeff", {}).get("ann", [])
        if isinstance(snpeff_ann, dict):
            snpeff_ann = [snpeff_ann]

        all_consequences = set()
        genes_for_row = []
        for annotation in snpeff_ann:
            effect = annotation.get("effect") or ""
            for term in str(effect).split("&"):
                if term:
                    all_consequences.add(term.strip())
            if annotation.get("gene_id"):
                genes_for_row.append(
                    {
                        "gene_id": annotation.get("gene_id"),
                        "genename": annotation.get("genename"),
                        "impact": annotation.get("putative_impact"),
                    }
                )

        unique_consequences = sorted(all_consequences)
        opengwas_var.at[i, "Variant_ID"] = variant_record.get("_id", "")
        opengwas_var.at[i, "Inferred_type"] = variant_type_info or "unknown"
        opengwas_var.at[i, "Unique_SnpEff_Consequences"] = unique_consequences

        multi_hot_vector = [0] * len(CONSEQUENCE_VOCABULARY)
        for idx, term in enumerate(CONSEQUENCE_VOCABULARY):
            if term in unique_consequences:
                multi_hot_vector[idx] = 1

        opengwas_var.at[i, "consequence_multi_hot_vector"] = multi_hot_vector
        opengwas_var.at[i, "Gene_Annotations"] = genes_for_row
        opengwas_var.at[i, "CADD_Score"] = str(variant_record.get("cadd", {}).get("phred"))

    return opengwas_var


def prepare_variant_features(opengwas_var: pd.DataFrame) -> pd.DataFrame:
    opengwas_var = opengwas_var.copy()
    opengwas_var["eaf"] = opengwas_var["eaf"].fillna(0.5)
    opengwas_var["CADD_Score"] = pd.to_numeric(opengwas_var["CADD_Score"], errors="coerce")
    median_cadd = opengwas_var["CADD_Score"].median()
    opengwas_var["CADD_Score"] = opengwas_var["CADD_Score"].fillna(median_cadd)
    opengwas_var["Feature_Missing_Bio"] = opengwas_var["Gene_Annotations"].apply(
        lambda x: 0 if isinstance(x, list) and len(x) > 0 else 1
    )

    temp_vectors = opengwas_var["consequence_multi_hot_vector"].apply(
        lambda x: x if isinstance(x, list) else [0] * len(CONSEQUENCE_VOCABULARY)
    )
    consequence_vectors = pd.DataFrame(temp_vectors.tolist(), index=opengwas_var.index)
    consequence_vectors.columns = [f"VAR_CON_{term}" for term in CONSEQUENCE_VOCABULARY]
    opengwas_var = pd.concat([opengwas_var, consequence_vectors], axis=1)

    type_one_hot = pd.get_dummies(opengwas_var["Inferred_type"].fillna("unknown"), prefix="VariantType").astype(int)
    for col in [f"VariantType_{v}" for v in VARIANT_TYPE_VOCABULARY]:
        if col not in type_one_hot.columns:
            type_one_hot[col] = 0
    type_one_hot = type_one_hot[[f"VariantType_{v}" for v in VARIANT_TYPE_VOCABULARY if f"VariantType_{v}" in type_one_hot.columns]]

    variant_object = pd.concat([opengwas_var, type_one_hot], axis=1)
    return variant_object


def build_variant_and_edge_tables(variant_object: pd.DataFrame, target_phenotype: Optional[str] = None) -> dict:
    variant_object = variant_object.copy()
    if "or_value" in variant_object.columns:
        df_vp_edges = variant_object[[
            "rsid",
            "population",
            "trait",
            "eaf",
            "beta",
            "or_value",
            "p",
        ]].copy()
    else:
        df_vp_edges = variant_object[[
            "rsid",
            "population",
            "trait",
            "eaf",
            "beta",
            "p",
        ]].copy()
    df_vp_edges["beta"] = pd.to_numeric(df_vp_edges["beta"], errors="coerce")
    df_vp_edges["beta_unified"] = df_vp_edges["beta"].fillna(np.log(pd.to_numeric(df_vp_edges["or_value"], errors="coerce")))
    max_val = df_vp_edges["beta_unified"].abs().replace(0, np.nan).max() or 1.0
    df_vp_edges["magnitude"] = df_vp_edges["beta_unified"].abs() / max_val
    df_vp_edges["direction"] = (df_vp_edges["beta_unified"] > 0).astype(float)
    df_vp_edges['p'] = df_vp_edges['p'].astype(float)
    df_vp_edges['p'] = df_vp_edges['p'].replace(0.0, 1e-300)

    df_vp_edges_refined = df_vp_edges[[
        "rsid",
        "population",
        "trait",
        "eaf",
        "p",
        "beta_unified",
        "magnitude",
        "direction",
    ]].copy()
    df_vp_edges_refined["target_ancestry"] = df_vp_edges_refined["population"].map(ANCESTRY_MAP)
    if target_phenotype:
        df_vp_edges_refined["target_phenotype"] = target_phenotype
    else:
        df_vp_edges_refined["target_phenotype"] = "HP:0005978"
    df_vp_edges_refined = df_vp_edges_refined.drop(columns=["population"])

    variant_node_cols = [
        col for col in variant_object.columns
        if col not in {
            "id", "build", "population", "sex", "ncase", "ncontrol", "trait",
            "chr", "position", "eaf", "beta", "se", "p", "n", "or_value",
            "ea", "nea", "source", "Variant_ID", "Inferred_type",
            "Unique_SnpEff_Consequences", "consequence_multi_hot_vector", "Gene_Annotations",
        }
    ]
    if "rsid" not in variant_node_cols:
        variant_node_cols.insert(0, "rsid")

    df_variant_node = variant_object[variant_node_cols].drop_duplicates(subset="rsid").set_index("rsid")

    variant_gene_edges = []
    unique_gene_ids = set()
    for _, row in variant_object.iterrows():
        rsid = row.get("rsid")
        gene_annotations = row.get("Gene_Annotations")
        if isinstance(gene_annotations, list):
            for ann in gene_annotations:
                gene_id = ann.get("gene_id")
                impact = ann.get("impact")
                if gene_id:
                    unique_gene_ids.add(gene_id)
                    variant_gene_edges.append(
                        {
                            "Source_Variant_rsid": rsid,
                            "Target_Gene_ID": gene_id,
                            "Edge_Feature_Impact": impact,
                        }
                    )

    df_genes = pd.DataFrame(sorted(unique_gene_ids), columns=["Gene_ID"]).drop_duplicates()
    df_vg_edges = pd.DataFrame(variant_gene_edges).drop_duplicates()

    return {
        "variant_node": df_variant_node,
        "variant_phenotype_edges": df_vp_edges_refined,
        "variant_gene_edges": df_vg_edges,
        "gene_nodes": df_genes,
        "variant_object": variant_object,
    }


def save_tables(tables: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["variant_node"].to_csv(output_dir / "variant_nodes.csv")
    tables["variant_phenotype_edges"].to_csv(output_dir / "vp_edges_refined.csv", index=False)
    tables["variant_gene_edges"].to_csv(output_dir / "vg_edges.csv", index=False)
    tables["gene_nodes"].to_csv(output_dir / "genes.csv", index=False)
    tables["variant_object"].to_csv(output_dir / "variant_object.csv", index=False)


def save_va_edges(variant_object: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_va_edges = variant_object[["rsid", "eaf", "target_ancestry", "target_phenotype"]].copy()
    df_va_edges["target_ancestry"] = df_va_edges["target_ancestry"].str.split('_').str.get(-1)
    # df_va_edges["edge_attr"] = df_va_edges["eaf"].apply(lambda eaf: np.array([eaf]))
    df_va_edges.to_csv(output_dir / "va_final.csv", index=False)


def main(args: argparse.Namespace) -> None:
    base_path = Path(args.output_dir)
    bearer_token = load_bearer_token()
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    print(f"\n")
    logging.info(f"Fetching all OpenGWAS studies for {args.trait_name}")
    data_all_gwas = request_json("https://api.opengwas.io/api/gwasinfo", headers)

    # Select studies using provided trait regex
    selected = select_studies(data_all_gwas, args.trait_pattern, exclude_phrases=args.exclude_phrase)
    if selected.empty:
        logging.error("No studies found after filtering with the provided trait pattern.")
        print(f"\n")
        return

    ids = selected["id"].dropna().astype(str).tolist()
    logging.info("Selected %d study IDs for trait pattern: %s", len(ids), args.trait_pattern)
    logging.info("Active exclusion phrases: %s", ", ".join(args.exclude_phrase or []))
    logging.info("Selected studies summary:")
    for _, row in selected[["id", "trait", "population", "ncase", "year"]].iterrows():
        logging.info(
            "- id=%s | trait=%s | population=%s | ncase=%s | year=%s",
            row.get("id"),
            row.get("trait"),
            row.get("population"),
            row.get("ncase"),
            row.get("year"),
        )

    if args.print_selected_studies_only:
        logging.info("Selected studies only mode enabled; stopping after printing selection summary.")
        return

    gwasinfo_df_all = fetch_selected_gwasinfo(ids, headers)
    if gwasinfo_df_all.empty:
        logging.error("No detailed GWAS info fetched.")
        return

    gwasinfo_var = gwasinfo_df_all[[col for col in ["id", "build", "population", "sex", "ncase", "ncontrol"] if col in gwasinfo_df_all.columns]]

    tophits_df_all = fetch_tophits(ids, headers)
    if tophits_df_all.empty:
        logging.error("No top hits returned from OpenGWAS.")
        return

    opengwas_var = gwasinfo_var.merge(tophits_df_all, on="id", how="inner")
    opengwas_var["source"] = "Open GWAS"

    # Load any user-specified catalogue files (can be zero or many)
    cat_df = load_catalogues(args.catalogues)
    if cat_df is not None:
        opengwas_var = pd.concat([opengwas_var, cat_df], ignore_index=True, axis=0)

    opengwas_var = annotate_variants(opengwas_var)
    variant_object = prepare_variant_features(opengwas_var)
    tables = build_variant_and_edge_tables(variant_object, target_phenotype=args.target_phenotype)
    save_tables(tables, base_path)
    save_va_edges(tables["variant_phenotype_edges"], base_path)

    logging.info("Done. Outputs written to %s", base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert notebook variant workflow into a script")
    parser.add_argument("--output-dir", default="pipeline/output/variants", help="Directory for CSV outputs")
    # parser.add_argument("--catalogues", nargs="*", default=[], help="Optional CSV catalogue files to include (space-separated list)")
    parser.add_argument("--catalogues", default=[], help="Optional CSV catalogue files to include (space-separated list)")
    parser.add_argument("--trait-pattern", default=r".*(type (2|ii)|t2d) diabetes.*", help="Regex pattern to match study trait (case-insensitive)")
    parser.add_argument("--trait-name", default=None, help="The name of the trait")
    parser.add_argument("--target-phenotype", default=None, help="HPO/phenotype ID to write into the variant-phenotype edge table")
    parser.add_argument("--exclude-phrase", dest="exclude_phrase", action="append", help="Phrase to exclude from traits; can be passed multiple times")
    parser.add_argument("--print-selected-studies-only", action="store_true", help="Print selected studies and stop before fetching further data")
    args = parser.parse_args()
    main(args)
