import os
import yaml
import shutil
import warnings
import pandas as pd
from pathlib import Path

# Suppress the specific pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="stopit")

# Helper functions to load configuration and resolve paths
def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f) or {}

def resolve_pipeline_dir() -> Path:
    working_dir = Path.cwd()
    candidates = [working_dir, working_dir / "pipeline"]
    for candidate in candidates:
        if (candidate / "code" / "variants.py").exists():
            return candidate
    return working_dir


# Load configuration and resolve paths
cfg = load_config()
pipeline_dir = resolve_pipeline_dir()

variants_cfg = cfg.get("variants", {})
tissues_cfg = cfg.get("tissues", {})
vgt_edges_cfg = cfg.get("vgt_edges", {})
gene_gene_cfg = cfg.get("gene_gene", {})
gene_sources_cfg = cfg.get("gene_sources_and_list", {})
gene_features_cfg = cfg.get("gene_features", {})
gene_features_imputation_cfg = cfg.get("gene_features_imputation", {})
variant_gene_edges_cfg = cfg.get("variant_gene_edges", {})
gene_tissue_edges_cfg = cfg.get('gene_tissue_edges', {})
gene_pathway_cfg = cfg.get('gene_pathway', {})
gene_phenotype_edges_cfg = cfg.get('gene_phenotype_edges', {})
phenotype_features_cfg = cfg.get('phenotype_features', {})
phenotype_phenotype_edges_cfg = cfg.get('phenotype_phenotype_edges', {})
drug_gene_edges_cfg = cfg.get('drug_gene_edges', {})
drug_phenotype_edges_cfg = cfg.get('drug_phenotype_edges', {})
tissue_phenotype_edges_cfg = cfg.get('tissue_phenotype_edges', {})
phenotype_mapping_cfg = cfg.get('phenotype_mapping', {})
phenotype_phenotype_edges_all_cfg = cfg.get('phenotype_phenotype_edges_all', {})

# 1. Extract configuration values for variants
variant_traits = variants_cfg.get("traits", [])
variant_trait_params = []

for trait_cfg in variant_traits:
    trait_name = trait_cfg.get("name", "trait")
    trait_id = trait_cfg.get("id", "")
    variant_trait_pattern = trait_cfg.get("trait_pattern", "")
    variant_catalogues = trait_cfg.get("catalogues_to_include", [])
    variant_output_dir = trait_cfg.get("output_dir", f"output/variants/{trait_name}")
    if not Path(variant_output_dir).is_absolute():
        variant_output_dir = str((pipeline_dir / variant_output_dir).resolve())

    exclusion_phrases = trait_cfg.get("exclusion_phrases", [])
    if isinstance(exclusion_phrases, dict):
        exclusion_phrases = [phrase for phrase, enabled in exclusion_phrases.items() if enabled]
    elif not isinstance(exclusion_phrases, list):
        exclusion_phrases = [str(exclusion_phrases)]

    catalogue_args = ""
    if variant_catalogues:
        catalogue_args = "--catalogues " + "::".join(str(x) for x in variant_catalogues)

    exclude_args = " ".join(f'--exclude-phrase "{phrase}"' for phrase in exclusion_phrases if phrase)
    # Per-trait setting overrides the global "variants.print_selected_studies_only" default
    print_selected_studies_only = trait_cfg.get(
        "print_selected_studies_only", variants_cfg.get("print_selected_studies_only", False)
    )
    print_only_flag = "--print-selected-studies-only" if print_selected_studies_only else ""

    variant_trait_params.append({
        "name": trait_name,
        "id": trait_id,
        "trait_pattern": variant_trait_pattern,
        "catalogue_args": catalogue_args,
        "exclude_args": exclude_args,
        "print_only_flag": print_only_flag,
        "output_dir": variant_output_dir,
    })

# Fallback for a legacy single-trait config block
if not variant_trait_params:
    variant_trait_params.append({
        "name": "default",
        "id": "",
        "trait_pattern": variants_cfg.get("trait_pattern", r".*(type (2|ii)|t2d) diabetes.*"),
        "catalogue_args": "--catalogues " + "::".join(str(x) for x in variants_cfg.get("catalogues_to_include", [])) if variants_cfg.get("catalogues_to_include") else "",
        "exclude_args": " ".join(
            f'--exclude-phrase "{phrase}"' for phrase, enabled in variants_cfg.get("exclusion_phrases", {}).items() if enabled
        ),
        "print_only_flag": "--print-selected-studies-only" if variants_cfg.get("print_selected_studies_only", False) else "",
        "output_dir": str((pipeline_dir / variants_cfg.get("output_dir", "output/variants/default")).resolve()),
    })

variant_trait_output_dirs = [p["output_dir"] for p in variant_trait_params] or [str((pipeline_dir / "output/variants").resolve())]

# Combined dir merges every trait's outputs so downstream rules aren't limited to just the first trait
variant_output_dir_combined = variants_cfg.get("combined_output_dir", "output/variants/combined")
if not Path(variant_output_dir_combined).is_absolute():
    variant_output_dir_combined = str((pipeline_dir / variant_output_dir_combined).resolve())

variant_shell_commands = []
for cfg in variant_trait_params:
    cmd = [f"python {pipeline_dir / 'code' / 'variants.py'}"]
    cmd.append(f'--output-dir "{cfg["output_dir"]}"')
    if cfg["catalogue_args"]:
        cmd.append(cfg["catalogue_args"])
    cmd.append(f'--trait-pattern "{cfg["trait_pattern"]}"')
    cmd.append(f'--trait-name "{cfg["name"]}"')
    cmd.append(f'--target-phenotype "{cfg["id"]}"')
    if cfg["exclude_args"]:
        cmd.append(cfg["exclude_args"])
    if cfg["print_only_flag"]:
        cmd.append(cfg["print_only_flag"])
    variant_shell_commands.append(" \\\n    ".join(cmd))
variant_shell_script = "\n".join(variant_shell_commands)

# 2. Extract configuration values for tissues
tissues_output_dir = tissues_cfg.get("output_dir", "output/tissues")
if not Path(tissues_output_dir).is_absolute():
    tissues_output_dir = str((pipeline_dir / tissues_output_dir).resolve())

# 3. Extract configuration values for V_G_T edges
vgt_output_dir = vgt_edges_cfg.get("output_dir", "output/vgt_edges")
if not Path(vgt_output_dir).is_absolute():
    vgt_output_dir = str((pipeline_dir / vgt_output_dir).resolve())

# 4. Extract configuration values for gene-gene edges
gene_gene_output_dir = gene_gene_cfg.get("output_dir", "output/gene_gene")
if not Path(gene_gene_output_dir).is_absolute():
    gene_gene_output_dir = str((pipeline_dir / gene_gene_output_dir).resolve())

cache_dir = gene_gene_cfg.get("cache_dir", "data/string_cache")
if not Path(cache_dir).is_absolute():
    cache_dir = str((pipeline_dir / cache_dir).resolve())

# 5. Extract configuration values for the gene sources and list step
gene_sources_output_dir = gene_sources_cfg.get("output_dir", "output/gene_sources_and_list")
if not Path(gene_sources_output_dir).is_absolute():
    gene_sources_output_dir = str((pipeline_dir / gene_sources_output_dir).resolve())

# 6. Extract configuration values for gene features and imputation
gene_features_output_dir = gene_features_cfg.get("output_dir", "output/gene_features")
if not Path(gene_features_output_dir).is_absolute():
    gene_features_output_dir = str((pipeline_dir / gene_features_output_dir).resolve())

vocab_size = gene_features_cfg.get("vocab_size", 75)

# 7. Extract configuration values for gene features imputation
gene_features_imputation_output_dir = gene_features_imputation_cfg.get("output_dir",  "output/gene_features")
if not Path(gene_features_imputation_output_dir).is_absolute():
    gene_features_imputation_output_dir = str((pipeline_dir / gene_features_imputation_output_dir).resolve())

# 8. Extract configuration values for variant-gene edges
variant_gene_edges_output_dir = variant_gene_edges_cfg.get("output_dir", "output/variant_gene_edges")
if not Path(variant_gene_edges_output_dir).is_absolute():
    variant_gene_edges_output_dir = str((pipeline_dir / variant_gene_edges_output_dir).resolve())

# 9. Extract configuration values for gene-tissue edges
gene_tissue_edges_output_dir = gene_tissue_edges_cfg.get("output_dir", "output/gene_tissue_edges")
if not Path(gene_tissue_edges_output_dir ).is_absolute():
    gene_tissue_edges_output_dir = str((pipeline_dir / gene_tissue_edges_output_dir).resolve())

# 10. Extract configuration values for gene-pathway edges
gene_pathway_output_dir = gene_pathway_cfg.get("output_dir", "output/gene_pathway")
if not Path(gene_pathway_output_dir).is_absolute():
    gene_pathway_output_dir = str((pipeline_dir / gene_pathway_output_dir).resolve())

# 11. Extract configuration values for gene-phenotype edges
gene_phenotype_output_dir = gene_phenotype_edges_cfg.get("output_dir", "output/gene_phenotype")
include_other_clinical_phenotypes = gene_phenotype_edges_cfg.get("include_other_clinical_phenotypes", False)
if not Path(gene_phenotype_output_dir).is_absolute():
    gene_phenotype_output_dir = str((pipeline_dir / gene_phenotype_output_dir).resolve())

# 12. Extract configuration values for phenotype features
phenotypes_prevalence_file = phenotype_features_cfg.get("phenotypes_prevalence_file", "data/phenotypes_prevalence.csv")
phenotype_features_output_dir = phenotype_features_cfg.get("output_dir", "output/phenotype_features")
if not Path(phenotype_features_output_dir).is_absolute():
    phenotype_features_output_dir = str((pipeline_dir / phenotype_features_output_dir).resolve())

# 13. Extract configuration values for phenotype-phenotype edges
phenotype_phenotype_edges_output_dir = phenotype_phenotype_edges_cfg.get("output_dir", "output/phenotype_features")
if not Path(phenotype_phenotype_edges_output_dir).is_absolute():
    phenotype_phenotype_edges_output_dir = str((pipeline_dir / phenotype_phenotype_edges_output_dir).resolve())

# 14. Extract configuration values for drug-gene edges
drug_gene_edges_output_dir = drug_gene_edges_cfg.get("output_dir", "output/drug_gene_edges")
if not Path(drug_gene_edges_output_dir).is_absolute():
    drug_gene_edges_output_dir = str((pipeline_dir / drug_gene_edges_output_dir).resolve())

# 15. Extract configuration values for drug-phenotype edges
phenotype_code_file = drug_phenotype_edges_cfg.get("phenotype_code_file", "data/phenotypes_prevalence.csv")
drug_phenotype_edges_output_dir = drug_phenotype_edges_cfg.get("output_dir", "output/drug_phenotype_edges")
if not Path(drug_phenotype_edges_output_dir).is_absolute():
    drug_phenotype_edges_output_dir = str((pipeline_dir / drug_phenotype_edges_output_dir).resolve())

# 16. Extract configuration values for tissue-phenotype edges
tissue_enrichment_map = tissue_phenotype_edges_cfg.get("enrichment_map", [])
tissue_enrichment_file_path = tissue_phenotype_edges_cfg.get("enrichment_file_path", [])
tissue_enrichment_map_args = "--tissue-enrichment-map " + "::".join(str(x) for x in tissue_enrichment_map)
tissue_phenotype_edges_output_dir = tissue_phenotype_edges_cfg.get("output_dir", "output/tissue_phenotype_edges")
if not Path(tissue_phenotype_edges_output_dir).is_absolute():
    tissue_phenotype_edges_output_dir = str((pipeline_dir /tissue_phenotype_edges_output_dir).resolve())

# 17. Extract configuration values for phenotype mapping
phenotype_phenotype_edges_ldsc_file = phenotype_mapping_cfg.get("phenotype_phenotype_edges_ldsc_file", "data/phenotype_phenotype_edges_ldsc.csv")
phenotype_mapping_output_dir = phenotype_mapping_cfg.get("output_dir", "output/phenotype_mapping")
if not Path(phenotype_mapping_output_dir).is_absolute():
    phenotype_mapping_output_dir = str((pipeline_dir / phenotype_mapping_output_dir).resolve())

#18. Extract configuration values for phenotype-phenotype edges all
phenotype_phenotype_edges_all_output_dir = phenotype_phenotype_edges_all_cfg.get("output_dir", "output/phenotype_mapping")
if not Path(phenotype_phenotype_edges_all_output_dir).is_absolute():
    phenotype_phenotype_edges_all_output_dir = str((pipeline_dir / phenotype_phenotype_edges_all_output_dir).resolve())
 

# Build command-line arguments for the variants script
# catalogue_args = ""
# if variant_catalogues:
#     catalogue_args = "--catalogues " + " ".join(str(x) for x in variant_catalogues)

# exclude_args = ""
# for phrase, enabled in variant_exclusion_map.items():
#     if enabled:
#         exclude_args += f' --exclude-phrase "{phrase}"'



# Define paths to the scripts in the pipeline
variants_script_path = pipeline_dir / "code" / "variants.py"
tissues_script_path = pipeline_dir / "code" / "tissues.py"
vgt_script_path = pipeline_dir / "code" / "vgt_edges.py"
gene_gene_script_path = pipeline_dir / "code" / "gene_gene_edges.py"
gene_sources_script_path = pipeline_dir / "code" / "gene_sources_and_list.py"
gene_features_script_path = pipeline_dir / "code" / "gene_features.py"
gene_features_imputation_script_path = pipeline_dir / "code" / "gene_imputation.py"
variant_gene_edges_script_path = pipeline_dir / "code" / "variant_gene_edges.py"
gene_tissue_edges_script_path = pipeline_dir / "code" / "gene_tissue_edges.py"
gene_pathway_script_path = pipeline_dir / "code" / "gene_pathways.py"
gene_phenotype_script_path = pipeline_dir / "code" / "gene_phenotype.py"
phenotype_features_script_path = pipeline_dir / "code" / "phenotype_features.py"
phenotype_phenotype_script_path = pipeline_dir / "code" / "phenotype_phenotype.py"
drug_gene_edges_script_path = pipeline_dir / "code" / "drug_gene.py"
drug_phenotype_edges_script_path = pipeline_dir / "code" / "drug_phenotype.py"
tissue_phenotype_edges_script_path = pipeline_dir / "code" / "tissue_phenotype.py"
phenotype_mapping_script_path = pipeline_dir / "code" / "phenotype_combination.py"
phenotype_phenotype_edges_all_script_path = pipeline_dir / "code" / "phenotype_phenotype_cui.py"

# Define Snakemake rules for the pipeline
rule variants:
    output:
        [directory(d) for d in variant_trait_output_dirs],
    params:
        script=str(variants_script_path),
        shell_script=variant_shell_script,
    shell:
        """
        {params.shell_script}
        """

rule combine_variant_outputs:
    input:
        variant_node=[f"{d}/variant_node.csv" for d in variant_trait_output_dirs],
        vg_edges=[f"{d}/vg_edges.csv" for d in variant_trait_output_dirs],
        vp_edges=[f"{d}/vp_edges_refined.csv" for d in variant_trait_output_dirs],
    output:
        variant_node=f"{variant_output_dir_combined}/variant_node.csv",
        vg_edges=f"{variant_output_dir_combined}/vg_edges.csv",
        vp_edges=f"{variant_output_dir_combined}/vp_edges_refined.csv",
    run:
        # Key columns that identify a "duplicate" row per combined file
        dedup_keys = {
            output.variant_node: ["rsid"],
            output.vg_edges: ["Source_Variant_rsid", "Target_Gene_ID"],
            output.vp_edges: ["rsid", "target_ancestry", "target_phenotype"],
        }
        for out_path, in_paths in (
            (output.variant_node, input.variant_node),
            (output.vg_edges, input.vg_edges),
            (output.vp_edges, input.vp_edges),
        ):
            combined = pd.concat([pd.read_csv(p) for p in in_paths], ignore_index=True)
            # These GWAS-catalogue columns aren't part of the variant_node schema
            combined = combined.drop(columns=["STRONGEST SNP-RISK ALLELE", "SNPS"], errors="ignore")
            # Prefer the most complete row when the same key appears in more than one trait
            completeness = combined.notna().sum(axis=1)
            combined = combined.loc[completeness.sort_values(ascending=False).index]
            combined = combined.drop_duplicates(subset=dedup_keys[out_path], keep="first")
            combined.to_csv(out_path, index=False)

rule tissues:
    output:
        tissue_node=f"{tissues_output_dir}/tissue_node.csv",
    params:
        script=tissues_script_path,
        out_dir=tissues_output_dir
    shell:
        """
        python {params.script} \
            --output-dir {params.out_dir}
        """

rule vgt_edges:
    input:
        variant_node=f"{variant_output_dir_combined}/variant_node.csv",
        tissue_node=f"{tissues_output_dir}/tissue_node.csv"
    output:
        vgt_edges=f"{vgt_output_dir}/variant_tissue_edges_final.csv"
    params:
        script=vgt_script_path,
        out_dir=vgt_output_dir
    shell:
        """
        python {params.script} \
            --variant-node {input.variant_node} \
            --tissue-node {input.tissue_node} \
            --output-dir {params.out_dir}
        """

rule gene_gene_edges:
    input:
        vg_edges=f"{variant_output_dir_combined}/vg_edges.csv",
        vgt_edges=f"{vgt_output_dir}/variant_tissue_edges_final.csv",
    output:
        gene_gene_edges=f"{gene_gene_output_dir}/df_ppi_final.csv",
        gene_list=f"{gene_gene_output_dir}/modifiers_and_VGT_gene_list.csv"
    params:
        script=gene_gene_script_path,
        cache_dir=cache_dir,
        out_dir=gene_gene_output_dir
    shell:
        """
        python {params.script} \
            --vg-edges {input.vg_edges} \
            --variant-tissue-edges {input.vgt_edges} \
            --cache-dir {params.cache_dir} \
            --output-dir {params.out_dir}
        """

rule gene_sources_and_list:
    input:
        modifiers_and_vgt_gene_list=f"{gene_gene_output_dir}/modifiers_and_VGT_gene_list.csv",
        ppi=f"{gene_gene_output_dir}/df_ppi_final.csv",
        vg_edges=f"{variant_output_dir_combined}/vg_edges.csv",
    output:
        gene_list=f"{gene_sources_output_dir}/df_gene_collapsed.csv",
    params:
        script=gene_sources_script_path,
        out_dir=gene_sources_output_dir
    shell:
        """
        python {params.script} \
            --modifiers-and-vgt-gene-list {input.modifiers_and_vgt_gene_list} \
            --ppi {input.ppi} \
            --vg-edges {input.vg_edges} \
            --output-dir {params.out_dir} 
        """

rule gene_features:
    input:
        unique_gene_list=f"{gene_sources_output_dir}/df_gene_collapsed.csv",
    output:
        gene_features_all=f"{gene_features_output_dir}/gene_features_all.csv",
        gene_node_final=f"{gene_features_output_dir}/gene_node_final.csv",
    params:
        script=gene_features_script_path,
        vocab_size=vocab_size,
        out_dir=gene_features_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-list {input.unique_gene_list} \
            --vocab-size {params.vocab_size} \
            --output-dir {params.out_dir} 
        """

rule gene_features_imputation:
    input:
        unique_gene_list=f"{gene_sources_output_dir}/df_gene_collapsed.csv",
        vg_edges=f"{variant_output_dir_combined}/vg_edges.csv",
        vgt_edges=f"{vgt_output_dir}/variant_tissue_edges_final.csv",
        gene_features_all=f"{gene_features_output_dir}/gene_features_all.csv"
    output:
        gene_features_imputed=f"{gene_features_imputation_output_dir}/gene_features_imputted.csv",
        genes_node_dedup_mapped=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv"
    params:
        script=gene_features_imputation_script_path,
        vocab_size=vocab_size,
        out_dir=gene_features_imputation_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-list {input.unique_gene_list} \
            --vg-edges {input.vg_edges} \
            --variant-tissue-edges {input.vgt_edges} \
            --gene-features {input.gene_features_all} \
            --vocab-size {params.vocab_size} \
            --output-dir {params.out_dir} 
        """

rule variant_gene_edges:
    input:
        unique_gene_node=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv",
        vg_edges=f"{variant_output_dir_combined}/vg_edges.csv",
        variant_tissue_edges=f"{vgt_output_dir}/variant_tissue_edges_final.csv",
    output:
        vg_edges_all=f"{variant_gene_edges_output_dir}/df_vg_master_all.csv",
        vg_edges_cleaned = f"{variant_gene_edges_output_dir}/df_vg_master_clean.csv"
    params:
        script=variant_gene_edges_script_path,
        out_dir=variant_gene_edges_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-node {input.unique_gene_node} \
            --vg-edges {input.vg_edges} \
            --variant-tissue-edges {input.variant_tissue_edges} \
            --output-dir {params.out_dir} 
        """

rule gene_tissue_edges:
    input:
        unique_gene_node=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv",
        tissue_node=f"{tissues_output_dir}/tissue_node.csv"
    output:
        gene_tissue_edges=f"{gene_tissue_edges_output_dir}/gene_tissue_edges_final.csv",
    params:
        script=gene_tissue_edges_script_path,
        out_dir=gene_tissue_edges_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-node {input.unique_gene_node} \
            --tissue-node {input.tissue_node} \
            --output-dir {params.out_dir} 
        """

rule gene_pathway_edges:
    input:
        unique_gene_node=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv",
    output:
        gene_pathway_edges=f"{gene_pathway_output_dir}/gene_pathway_edges.csv",
        gene_pathway_nodes_raw=f"{gene_pathway_output_dir}/pathway_nodes_raw.csv",
        pathway_nodes=f"{gene_pathway_output_dir}/pathway_nodes.csv"
    params:
        script=gene_pathway_script_path,
        out_dir=gene_pathway_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-node {input.unique_gene_node} \
            --output-dir {params.out_dir}
        """

rule gene_phenotype_edges:
    input:
        unique_gene_node=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv",
        variant_phenotype_edges=f"{variant_output_dir_combined}/vp_edges_refined.csv",
        variant_gene_edges=f"{variant_gene_edges_output_dir}/df_vg_master_clean.csv",
    output:
        gene_phenotype_edges=f"{gene_phenotype_output_dir}/gene_phenotype_edges.csv",
        new_phenotypes=f"{gene_phenotype_output_dir}/new_phenotypes_from_gp_edges.csv",
    params:
        script=gene_phenotype_script_path,
        include_other_clinical_phenotypes=include_other_clinical_phenotypes,
        out_dir=gene_phenotype_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-node {input.unique_gene_node} \
            --variant-phenotype-edges {input.variant_phenotype_edges} \
            --variant-gene-edges {input.variant_gene_edges} \
            --include-clinical-phenotypes {params.include_other_clinical_phenotypes} \
            --output-dir {params.out_dir}
        """

rule build_phenotype_features: 
    input:
        phenotypes_with_prevalence=phenotypes_prevalence_file,
        new_phenotypes=f"{gene_phenotype_output_dir}/new_phenotypes_from_gp_edges.csv"
    output:
        phenotype_features=f"{phenotype_features_output_dir}/phenotype_features.csv",
        ancestry_node=f"{phenotype_features_output_dir}/ancestry_nodes.csv",
        ancestry_phenotype_edges=f"{phenotype_features_output_dir}/ancestry_phenotype_edges.csv"
    params:
        script=phenotype_features_script_path,
        out_dir=phenotype_features_output_dir
    shell:
        """
        python {params.script} \
            --phenotypes-with-prevalence {input.phenotypes_with_prevalence} \
            --new-phenotypes-from-gp-edges {input.new_phenotypes} \
            --output-dir {params.out_dir}
        """

rule phenotype_phenotype_edges:
    input:
        phenotype_features=f"{phenotype_phenotype_edges_output_dir}/phenotype_features.csv"
    output:
        phenotype_phenotype_edges=f"{phenotype_phenotype_edges_output_dir}/phenotype_phenotype_edges.csv"
    params:
        script=phenotype_phenotype_script_path,
        out_dir=phenotype_phenotype_edges_output_dir
    shell:
        """
        python {params.script} \
            --phenotype-features {input.phenotype_features} \
            --output-dir {params.out_dir}
        """

rule build_drug_gene_edges:
    input:
        unique_gene_node=f"{gene_features_imputation_output_dir}/genes_node_dedup_mapped.csv",
    output:
        drug_gene_edges=f"{drug_gene_edges_output_dir}/drug_gene_edges.csv",
        drug_gene_edges_final=f"{drug_gene_edges_output_dir}/drug_gene_edges_final.csv",
        drug_indications=f"{drug_gene_edges_output_dir}/drug_indications.csv",
        drug_smiles=f"{drug_gene_edges_output_dir}/drug_smiles.csv",
        drug_features=f"{drug_gene_edges_output_dir}/all_drug_features.csv"
    params:
        script=drug_gene_edges_script_path,
        out_dir=drug_gene_edges_output_dir
    shell:
        """
        python {params.script} \
            --unique-gene-node {input.unique_gene_node} \
            --output-dir {params.out_dir}
        """

rule build_drug_phenotype_edges:
    input:
        phenotype_features=f"{phenotype_features_output_dir}/phenotype_features.csv",
        drug_indications=f"{drug_gene_edges_output_dir}/drug_indications.csv",
        phenotype_code_file=phenotype_code_file
    output:
        drug_causes_phenotype=f"{drug_phenotype_edges_output_dir}/drug_causes_phenotype_edges.csv",
        side_effects_features=f"{drug_phenotype_edges_output_dir}/side_effects_features.csv",
        side_effect_names_and_codes=f"{drug_phenotype_edges_output_dir}/side_effect_names_and_codes.csv",
        drug_treats_phenotype=f"{drug_phenotype_edges_output_dir}/drug_treats_phenotype_edges.csv"
    params:
        script=drug_phenotype_edges_script_path,
        out_dir=drug_phenotype_edges_output_dir
    shell:
        """
        python {params.script} \
            --phenotype-features {input.phenotype_features} \
            --drug-indications {input.drug_indications} \
            --phenotype-code-file {input.phenotype_code_file} \
            --output-dir {params.out_dir}
        """

rule build_tissue_phenotype_edges:
    input:
        phenotype_features=f"{phenotype_features_output_dir}/phenotype_features.csv",
        phenotypes_with_prevalence=phenotypes_prevalence_file,
        tissue_enrichment_file=tissue_enrichment_file_path
    output:
        tissue_enrichment_phenotype=f"{tissue_phenotype_edges_output_dir}/tissue_phenotype_edges.csv"
    params:
        script=tissue_phenotype_edges_script_path,
        tissue_enrichment_map_args=tissue_enrichment_map_args,
        out_dir=tissue_phenotype_edges_output_dir
    shell:
        """
        python {params.script} \
            --phenotype-features {input.phenotype_features} \
            --phenotypes-with-prevalence {input.phenotypes_with_prevalence} \
            --tissue-enrichment-file {input.tissue_enrichment_file} \
            {params.tissue_enrichment_map_args} \
            --output-dir {params.out_dir}
        """

rule build_phenotype_mapping:
    input:
        phenotype_features=f"{phenotype_features_output_dir}/phenotype_features.csv",
        phenotypes_with_prevalence=phenotypes_prevalence_file,
        side_effect_names_and_codes=f"{drug_phenotype_edges_output_dir}/side_effect_names_and_codes.csv",
        drug_causes_phenotype=f"{drug_phenotype_edges_output_dir}/drug_causes_phenotype_edges.csv",
        drug_treats_phenotype=f"{drug_phenotype_edges_output_dir}/drug_treats_phenotype_edges.csv",
        gene_phenotype_edges=f"{gene_phenotype_output_dir}/gene_phenotype_edges.csv",
        ancestry_phenotype_edges=f"{phenotype_features_output_dir}/ancestry_phenotype_edges.csv",
        phenotype_phenotype_edges_lin=f"{phenotype_phenotype_edges_output_dir}/phenotype_phenotype_edges.csv",
        phenotype_phenotype_edges_ldsc=phenotype_phenotype_edges_ldsc_file,
        tissue_phenotype_edges=f"{tissue_phenotype_edges_output_dir}/tissue_phenotype_edges.csv",
        variant_phenotype_edges=f"{variant_output_dir_combined}/vp_edges_refined.csv"
    output:
        phenotype_features=f"{phenotype_mapping_output_dir}/phenotype_features_cui.csv",
        ancestry_phenotype_edges=f"{phenotype_mapping_output_dir}/ancestry_phenotype_edges_cui.csv",
        drug_treats_phenotype=f"{phenotype_mapping_output_dir}/drug_treats_phenotype_edges_cui.csv",
        drug_causes_phenotype=f"{phenotype_mapping_output_dir}/drug_causes_phenotype_edges_cui.csv",
        gene_phenotype_edges=f"{phenotype_mapping_output_dir}/gene_phenotype_edges_cui.csv",
        phenotype_phenotype_edges_lin=f"{phenotype_mapping_output_dir}/phenotype_phenotype_edges_lin_cui.csv",
        phenotype_phenotype_edges_ldsc=f"{phenotype_mapping_output_dir}/phenotype_phenotype_edges_ldsc_cui.csv",
        tissue_phenotype_edges=f"{phenotype_mapping_output_dir}/tissue_phenotype_edges_cui.csv",
        variant_phenotype_edges=f"{phenotype_mapping_output_dir}/variant_phenotype_edges_cui.csv",
        clinical_outcomes_final=f"{phenotype_mapping_output_dir}/clinical_outcomes_final_cui.csv"
    params:
        script=phenotype_mapping_script_path,
        out_dir=phenotype_mapping_output_dir
    shell:
        """
        python {params.script} \
            --phenotype-features {input.phenotype_features} \
            --phenotypes-with-prevalence {input.phenotypes_with_prevalence} \
            --side-effect-names-and-codes {input.side_effect_names_and_codes} \
            --drug-causes-phenotype {input.drug_causes_phenotype} \
            --drug-treats-phenotype {input.drug_treats_phenotype} \
            --ancestry-phenotype-edges {input.ancestry_phenotype_edges} \
            --gene-phenotype-edges {input.gene_phenotype_edges} \
            --phenotype-phenotype-edges-lin {input.phenotype_phenotype_edges_lin} \
            --phenotype-phenotype-edges-ldsc {input.phenotype_phenotype_edges_ldsc} \
            --tissue-phenotype-edges {input.tissue_phenotype_edges} \
            --variant-phenotype-edges {input.variant_phenotype_edges} \
            --output-dir {params.out_dir}
        """

rule phenotype_phenotype_edges_cui_all:
    input:
        phenotype_features=f"{phenotype_mapping_output_dir}/phenotype_features_cui.csv",
    output:
        phenotype_phenotype_edges=f"{phenotype_mapping_output_dir}/phenotype_phenotype_lin_cui_all.csv"
    params:
        script=phenotype_phenotype_edges_all_script_path,
        out_dir=phenotype_mapping_output_dir
    shell:
        """
        python {params.script} \
            --phenotype-features {input.phenotype_features} \
            --output-dir {params.out_dir}
        """