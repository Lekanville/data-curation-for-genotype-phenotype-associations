# Genotype–Phenotype Data Curation Pipeline

This project builds a multi-omics graph and phenotype association pipeline from GWAS variant data, genes, tissues, pathways, drugs, and phenotype relationships. It is implemented as a Snakemake workflow and is configured through `config.yaml`.

The pipeline stages include:

- variant selection and filtering from GWAS catalog inputs
- tissue and gene-level graph construction
- gene–variant, gene–tissue, gene–pathway, and gene–phenotype edges
- phenotype feature generation and phenotype–phenotype relationships
- drug–gene and drug–phenotype edge generation
- phenotype mapping and final downstream integration

---

## Table of contents

- [Project overview](#project-overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
- [Understanding the outputs](#understanding-the-outputs)
- [Data expectations](#data-expectations)
- [Common issues](#common-issues)
- [Project structure](#project-structure)
- [Recommended workflow](#recommended-workflow)
- [License](#license)
- [Maintainer notes](#maintainer-notes)

---

## Project overview

The workflow is driven by `Snakefile` and `config.yaml`. The main logic is written in Python scripts under `code/`, and the generated datasets are written to `output/`.

The default workflow is configured for a Type 2 diabetes trait, but the `variants.traits` section in `config.yaml` can be modified to add or remove traits.

---

## Requirements

You will need:

- Linux or WSL-like environment
- Conda or Mamba installed
- Python 3 environment managed by the repository config
- Access to the data files under `data/`
- Required: a valid IEU OpenGWAS bearer token in `BEARER_TOKEN` for all variant/GWAS queries
- Required: a valid UMLS API key in `UMLS_API_KEY` for phenotype mapping and UMLS-based lookups

Both API keys are required for the pipeline to run successfully; the OpenGWAS bearer token is not optional.

The repository includes a Conda environment file at `environment.yml`.

---

## Setup

1. Clone the repository and move into the project root:

   ```bash
   git clone <repo-url>
   cd pipeline
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate gat_data_pipeline
   ```

4. Define the required API credentials in a `.env` file at the project root before running the pipeline:

   ```env
   BEARER_TOKEN="your_ieu_opengwas_bearer_token_here"
   UMLS_API_KEY="your_umls_api_key_here"
   ```

   The project loads these values with `python-dotenv`, so they should be stored in `.env` rather than only exported in the shell. `BEARER_TOKEN` is required for calls to the IEU OpenGWAS API used by the variant workflow. `UMLS_API_KEY` is required for phenotype mapping and UMLS-based conversion steps.

5. Confirm the project is ready:

   ```bash
   python --version
   snakemake --help
   ```

> If `snakemake` is not found after activating the environment, run `conda activate gat_data_pipeline` again and verify the environment was created successfully.

---

## Configuration

The pipeline settings live in `config.yaml`.

### Main configuration sections

- `variants`: trait definitions, catalogue files, exclusion phrases, and output directories
- `tissues`: tissue node generation settings
- `vgt_edges`: variant–tissue edge output directory
- `gene_gene`: STRING cache and output dir
- `gene_features`: gene feature generation settings
- `gene_phenotype_edges`: include other clinical phenotype options
- `phenotype_features`: prevalence file path
- `drug_gene_edges`: drug gene output dir
- `drug_phenotype_edges`: phenotype code reference
- `tissue_phenotype_edges`: enrichment mapping config
- `phenotype_mapping`: LDSC phenotype relationship files

### Example trait configuration

```yaml
variants:
  traits:
    - name: type_2_diabetes
      id: HP:0005978
      trait_pattern: '.*(type (2|ii)|t2d) diabetes.*'
      catalogues_to_include: [data/datasets/T2D/AFR_gwas_t2d.csv, data/datasets/T2D/GME_gwas_t2d.csv]
      exclusion_phrases: ["medication for", "self-reported", "do not have", "ever had diabetes", "gestational diabetes", "adjusted"]
      output_dir: output/variants/type_2_diabetes
```

### Important notes

- Add or remove traits in `variants.traits` to control which diseases are processed.
- `trait_pattern` is a regex used to match study descriptions.
- `catalogues_to_include` can point to local CSV files in data/datasets/.... This is optional and can be used to add missing ancestry-specific GWAS data for particular phenotypes when OpenGWAS coverage is incomplete. You can leave it as an empty list when it is not needed.
- `output_dir` is created automatically by the pipeline.
- `print_selected_studies_only` can be set to limit printed study selection outputs during filtering.

---

## Running the pipeline

The pipeline is orchestrated with Snakemake.

### Dry run

```bash
snakemake -n -p --cores 1
```

This prints the jobs that would run without executing them. It is useful for checking the DAG and validating the config.

### Full pipeline run

```bash
snakemake --cores 1
```

You can increase the core count if your machine supports it:

```bash
snakemake --cores 8
```

### Run a specific rule

If you want to execute only part of the workflow, you can target a rule name:

```bash
snakemake gene_features --cores 1
```

The exact rule names defined in the workflow are:

- `variants`
- `combine_variant_outputs`
- `tissues`
- `vgt_edges`
- `gene_gene_edges`
- `gene_sources_and_list`
- `gene_features`
- `gene_features_imputation`
- `variant_gene_edges`
- `gene_tissue_edges`
- `gene_pathway_edges`
- `gene_phenotype_edges`
- `build_phenotype_features`
- `phenotype_phenotype_edges`
- `build_drug_gene_edges`
- `build_drug_phenotype_edges`
- `build_tissue_phenotype_edges`
- `build_phenotype_mapping`
- `phenotype_phenotype_edges_cui_all`

---

## Understanding the outputs

Most results are written to directories under `output/`.

Key outputs include:

- `output/variants/` — variant extraction and refined phenotype-linked files
- `output/variants/combined/` — merged trait-level variant tables
- `output/tissues/` — tissue node data
- `output/vgt_edges/` — variant–tissue edges
- `output/gene_gene/` — gene-gene interaction data
- `output/gene_sources_and_list/` — collapsed gene lists
- `output/gene_features/` — gene features and final gene node outputs
- `output/variant_gene_edges/` — variant–gene relationships
- `output/gene_tissue_edges/` — gene–tissue edges
- `output/gene_pathway/` — pathway nodes and pathways connected to genes
- `output/gene_phenotype/` — gene–phenotype edges and new phenotypes
- `output/phenotype_features/` — phenotype features and ancestry mappings
- `output/drugs_nodes_and_edges/` — drug and gene relationships
- `output/drug_phenotype_edges/` — drug phenotype edges and side effects
- `output/tissue_phenotype_edges/` — tissue–phenotype enrichment edges
- `output/phenotype_mapping/` — mapped phenotype outputs and final CUI-integrated files

---

## Data expectations

The project expects the following inputs to be available in or referenced from `data/`:

- `data/phenotypes_prevalence.csv` — required phenotype prevalence metadata used by phenotype feature generation
- `data/enrichments/...` — required enrichment files for tissue–phenotype relationship analysis
- `data/phenotype_phenotype_edges_ldsc.csv` — optional but recommended LDSC phenotype edge input for similar-phenotype relationships
- `data/datasets/...` — optional external GWAS catalogue CSV files only when you want to include ancestry-specific or missing GWAS data not available through IEU OpenGWAS. These are configured under `variants.traits[].catalogues_to_include` in `config.yaml`

The following resources are not expected to be pre-populated in the repo and are created or downloaded during runtime:

- `data/string_cache/` — populated by the STRING-based gene network workflow
- `obo/hp.obo` — downloaded or generated as needed for ontology support

If your dataset layout differs, update the file paths in `config.yaml` before running the workflow.

---

## Common issues

### 1. `snakemake: command not found`

Ensure the environment is active:

```bash
conda activate gat_data_pipeline
which snakemake
```

### 2. Missing `BEARER_TOKEN` or `UMLS_API_KEY`

This project uses both the IEU OpenGWAS API and UMLS during the workflow. Define both variables in a `.env` file at the project root:

```env
BEARER_TOKEN="..."
UMLS_API_KEY="..."
```

The OpenGWAS bearer token is required for the variant-selection step and is not optional.

### 3. No data files found

Check that the paths in `config.yaml` point to real files in `data/` and that the dataset folders exist.

### 4. Large or slow workflow

This pipeline can be computationally heavy. Use a dry run first and then run with multiple cores if your environment allows it.

---

## Project structure

```text
pipeline/
├── Snakefile
├── config.yaml
├── environment.yml
├── README.md
├── code/
│   ├── variants.py
│   ├── tissues.py
│   ├── vgt_edges.py
│   ├── gene_gene_edges.py
│   ├── gene_sources_and_list.py
│   ├── gene_features.py
│   ├── gene_imputation.py
│   ├── variant_gene_edges.py
│   ├── gene_tissue_edges.py
│   ├── gene_pathways.py
│   ├── gene_phenotype.py
│   ├── phenotype_features.py
│   ├── phenotype_phenotype.py
│   ├── drug_gene.py
│   ├── drug_phenotype.py
│   ├── tissue_phenotype.py
│   ├── phenotype_combination.py
│   ├── phenotype_phenotype_cui.py
│   └── tools/
├── data/
├── obo/
├── output/
├── notebooks/
└── .gitignore
```

---

## Recommended workflow

For a first run, use this sequence:

```bash
conda env create -f environment.yml
conda activate gat_data_pipeline

# create a .env file in the project root with:
# BEARER_TOKEN="your_ieu_opengwas_bearer_token"
# UMLS_API_KEY="your_umls_api_key"

snakemake -n -p --cores 1
snakemake --cores 1
```

This is the safest way to validate the config and then execute the full pipeline.

---

## License

This project does not currently declare a license in the repository root. If you intend to share or publish it externally, add an appropriate license file before distribution.

---

## Maintainer notes

This workflow is best customized by editing `config.yaml` before running Snakemake. Most changes to trait scope, dataset location, or output naming can be made there without modifying the Python pipeline scripts.
