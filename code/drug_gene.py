import io
import time
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from chembl_webresource_client.new_client import new_client


def normalize_chembl_id(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.startswith("CHEMBL:"):
        return "CHEMBL" + value.split("CHEMBL:", 1)[1]
    if value.startswith("CHEMBL"):
        return value
    return np.nan


def get_molecule_chembl_ids(names_to_search):
    """
    Maps a list of drug names to their corresponding ChEMBL IDs using the ChEMBL API.
    Returns a dictionary mapping drug names to ChEMBL IDs.
    """
    molecule_map = {}
    print(f"🔍 Mapping {len(names_to_search)} names...")

    for name in tqdm(names_to_search):
        try:
            res = new_client.molecule.search(name).only(['molecule_chembl_id', 'pref_name'])
            if res and len(res) > 0:
                chembl_id = res[0].get('molecule_chembl_id')
                if chembl_id:
                    molecule_map[str(name).strip().upper()] = chembl_id
        except Exception:
            continue

    return molecule_map

def get_drug_indications(all_target_ids):
    """
    Fetches drug indications for a list of ChEMBL IDs using the ChEMBL API.
    Returns a dataframe of drug indications.
    """
    all_indications = []
    chunk_size = 20 # Small chunks prevent the 8-hour stall

    print(f"🚀 Fetching indications for {len(all_target_ids)} unique IDs...")

    for i in tqdm(range(0, len(all_target_ids), chunk_size)):
        id_chunk = all_target_ids[i:i + chunk_size]
        try:
            # We query by IDs, NOT names, because the API is more stable this way
            inds = new_client.drug_indication.filter(molecule_chembl_id__in=id_chunk)
            for ind in inds:
                all_indications.append({
                    'molecule_chembl_id': ind.get('molecule_chembl_id'),
                    'mesh_id': ind.get('mesh_id'),
                    'efo_id': ind.get('efo_id'),
                    'disease_name': ind.get('mesh_heading')
                })
        except Exception as e:
            continue 

    df_indications = pd.DataFrame(all_indications)
    return df_indications


def get_molecule_structures(all_target_ids):
    """
    Fetches molecular structures (SMILES) for a list of ChEMBL IDs using the ChEMBL API.
    Returns a dataframe of ChEMBL IDs and their corresponding SMILES.
    """
    target_ids = list(all_target_ids) 
    molecule_data = []
    batch_size = 100

    print(f"🚀 Fetching structures for {len(target_ids)} molecules...")

    for i in tqdm(range(0, len(target_ids), batch_size)):
        chunk = target_ids[i:i + batch_size]
        
        # Query the molecule endpoint specifically for structures
        res = new_client.molecule.filter(molecule_chembl_id__in=chunk).only([
            'molecule_chembl_id', 
            'molecule_structures'
        ])
        
        for r in res:
            structs = r.get('molecule_structures')
            # We prefer 'canonical_smiles'
            smiles = structs.get('canonical_smiles') if structs else None
            
            molecule_data.append({
                'molecule_chembl_id': r['molecule_chembl_id'],
                'smiles': smiles
            })

    df_smiles = pd.DataFrame(molecule_data)

    return df_smiles


def smiles_to_fp_modern(smiles):
    """
    Modern ECFP4 generation using the MorganGenerator API.
    """
    # Initialize the generator ONCE outside the function for speed
    # This "factory" now knows to always produce 1024-bit ECFP4
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    if pd.isna(smiles) or smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # FIX: Pass the molecule 'mol', not the generator itself
            return np.array(fp_gen.GetFingerprint(mol))
    except Exception:
        return None


def finalize_features(row):
    fingerprint = row.get('fingerprint')

    if fingerprint is None:
        return np.zeros(1025, dtype=float)

    if isinstance(fingerprint, np.ndarray):
        if fingerprint.size == 0:
            return np.zeros(1025, dtype=float)
        try:
            fp = fingerprint.astype(float).reshape(-1)
        except Exception:
            return np.zeros(1025, dtype=float)
    else:
        try:
            if pd.isna(fingerprint):
                return np.zeros(1025, dtype=float)
            fp = np.asarray(fingerprint, dtype=float).reshape(-1)
        except Exception:
            return np.zeros(1025, dtype=float)

    if fp.size < 1024:
        fp = np.pad(fp, (0, 1024 - fp.size), constant_values=0.0)
    elif fp.size > 1024:
        fp = fp[:1024]

    return np.concatenate(([1.0], fp))


def build_drug_gene_data(unique_gene_node: Path, output_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Builds the drug-gene edges and gene nodes dataframes.
    Returns the paths to the output files.
    """
    # Load the unique gene node data
    genes_node_full = pd.read_csv(unique_gene_node)

    base_url = "https://dgidb.org/data/latest/interactions.tsv"
    response = requests.get(base_url)
    response.raise_for_status()
        
    # Read the file. We must skip lines starting with '#' (metadata)
    df_drug = pd.read_csv(io.StringIO(response.text), sep='\t', comment='#', header=None, low_memory = False)
    df_drug.columns = df_drug.loc[0].tolist()
    df_drug.drop(0, axis = 0, inplace = True)
    df_drug.reset_index(drop = True, inplace = True)

    gene_list = genes_node_full["Gene_Symbol"].unique()

    df_drug_genes = df_drug[df_drug['gene_name'].isin(gene_list)]
    print(f"Found {len(df_drug_genes)} drug-gene interactions for the provided gene list.")

    # Filter only the standardized columns
    df_drug_gene_edges = df_drug_genes[[
        'drug_name',       # Standardized Drug Node
        'gene_name',       # Standardized Gene Node
        'interaction_type',# Edge Label
        'interaction_score',# Edge Weight
    ]].copy()

    # Map to GAT structure
    df_drug_gene_edges.columns = ['source', 'target', 'interaction_type', 'weight']

    # First, remove rows where the Drug name itself is missing
    df_drug_gene_edges.dropna(subset=['source', 'target'], inplace=True)

    # Fill missing interaction types (from our previous step)
    df_drug_gene_edges['interaction_type'] = df_drug_gene_edges['interaction_type'].fillna("undetermined_interaction")

    # Handle null weights: Convert to numeric and fill NaNs with a baseline
    df_drug_gene_edges['weight'] = pd.to_numeric(df_drug_gene_edges['weight'], errors='coerce')
    df_drug_gene_edges['weight'] = df_drug_gene_edges['weight'].fillna(0.1)


    # Normalize the 'weight' column
    # Log-transform to handle the long tail of publication counts smoothly
    df_drug_gene_edges['weight_log'] = np.log10(df_drug_gene_edges['weight'] + 1)

    # Then normalize safely
    scaler = MinMaxScaler()
    df_drug_gene_edges['weight_norm'] = scaler.fit_transform(df_drug_gene_edges[['weight_log']])

    # Consolidate nans into the existing 'other/unknown' class
    # df_drug_gene_edges['interaction_type'] = df_drug_gene_edges['interaction_type'].fillna('other/unknown')

    # One-Hot Encode 'interaction_type'
    # This creates columns: interaction_type_inhibitor, interaction_type_blocker, etc.
    df_encoded = pd.get_dummies(df_drug_gene_edges, columns=['interaction_type'], prefix='type')

    # Convert boolean columns to floats (0.0 or 1.0)
    type_cols = [col for col in df_encoded.columns if col.startswith('type_')]
    df_encoded[type_cols] = df_encoded[type_cols].astype(float)

    # The final 'edge_attr' for the GAT will be a combination of these new columns
    # feature_cols = ['weight_norm'] + [col for col in df_encoded.columns if col.startswith('type_')]
    # edge_features = df_encoded[feature_cols].values

    # The drugs and genes
    df_drug_gene = df_encoded.copy()
    drugs = df_drug_gene["source"].unique()
    # genes = df_drug_gene["target"].unique()

    # --- STEP 1: CLEAN PREFIXES ---
    chembl_ids_ready = []
    for drug_name in drugs:
        normalized = normalize_chembl_id(drug_name)
        if isinstance(normalized, str) and normalized.startswith("CHEMBL"):
            chembl_ids_ready.append(normalized)
    chembl_ids_ready = list(dict.fromkeys(chembl_ids_ready))

    names_to_search = [str(d).strip() for d in drugs if not str(normalize_chembl_id(d)).startswith("CHEMBL")]

    # --- STEP 2: STABLE MAPPING ---
    molecule_map = get_molecule_chembl_ids(names_to_search)

    # lookup_map = {str(k).upper(): v for k, v in molecule_map.items()}
    # source_values = df_drug_gene["source"].astype(str).str.upper()
    # normalized_ids = source_values.apply(lambda x: x if x.startswith("CHEMBL") else np.nan)
    # mapped_ids = source_values.map(lookup_map)
    # df_drug_gene["chembl_id"] = normalized_ids.fillna(mapped_ids)
    
    df_drug_gene["chembl_id"] = df_drug_gene["source"].str.split(":").str[1].fillna(df_drug_gene["source"].map(molecule_map))

    df_drug_gene_final = df_drug_gene.copy()
    df_drug_gene_final["source"] = df_drug_gene_final["chembl_id"]
    df_drug_gene_final.drop("chembl_id", axis = 1, inplace = True)
    cols = df_drug_gene_final.columns
    updated_cols = [c.replace(" ", "_").replace("/", "_").replace("-", "_") for c in cols]
    df_drug_gene_final.columns = updated_cols

    # CRITICAL VERIFICATION: Ensure we don't have a collision
    unique_ids = len(set(molecule_map.values()))
    print(f"✅ Mapped {len(molecule_map)} names to {unique_ids} unique IDs.")

    # --- STEP 3: CONSOLIDATE ---
    all_target_ids = list(set(chembl_ids_ready + list(molecule_map.values())))

    # --- STEP 4: FETCH INDICATIONS (The high-speed part) ---
    df_indications = get_drug_indications(all_target_ids)
    # print(df_indications.head())
    print(f"✅ Finished! Total indications found: {len(df_indications)}")

    # --- STEP 5: FETCH MOLECULAR STRUCTURES ---
    df_smiles = get_molecule_structures(all_target_ids)
    df_smiles.rename(columns={"molecule_chembl_id":"chembl_id"}, inplace = True)
    # Check for missing values (biologicals/large polymers often lack SMILES)
    missing_count = df_smiles['smiles'].isna().sum()
    found_count = len(df_smiles[~df_smiles['smiles'].isna()])
    print(f"✅ Finished! Found {found_count} SMILES. (Missing: {missing_count})")

    #  --- STEP 6: GENERATE FINGERPRINTS ---
    # Use the modern ECFP4 generator for speed and accuracy
    print("🔬 Generating ECFP4 fingerprints for the SMILES...")
    df_smiles['fingerprint'] = df_smiles['smiles'].apply(smiles_to_fp_modern)

    #  --- STEP 7: MERGE FEATURES ---
    # Note: Ensure the column name matches your 'all_target_ids' dataframe (molecule_chembl_id or chembl_id)
    all_drug_features = pd.DataFrame({'chembl_id': all_target_ids})
    all_drug_features = all_drug_features.merge(df_smiles, on='chembl_id', how='left')

    # --STEP 7: Create the 1025-dimension feature vector [Indicator + 1024 FP] ---
    all_drug_features['node_feature_vector'] = [
        finalize_features(row) for _, row in all_drug_features.iterrows()
    ]
    print(f"✅ Created 1025-bit feature vectors for {len(all_drug_features)} drug nodes.")

    # Save the final merged edges to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_gene_edges_dir = output_dir / "drug_gene_edges.csv"
    df_drug_gene_final_dir = output_dir / "drug_gene_edges_final.csv"
    df_indications_dir = output_dir / "drug_indications.csv"
    df_smiles_dir = output_dir / "drug_smiles.csv"
    all_drug_features_dir = output_dir / "all_drug_features.csv"

    df_encoded.to_csv(drug_gene_edges_dir, index=False)
    df_drug_gene_final.to_csv(df_drug_gene_final_dir, index=False)
    df_indications.to_csv(df_indications_dir, index=False)
    df_smiles.to_csv(df_smiles_dir, index=False)
    all_drug_features.to_csv(all_drug_features_dir, index=False)

    return drug_gene_edges_dir, df_drug_gene_final_dir, df_indications_dir, df_smiles_dir, all_drug_features_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Get drug-gene data")
    parser.add_argument("--unique-gene-node", default="genes_node_dedup_mapped.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--output-dir", default="output/drug_gene", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = build_drug_gene_data(
        unique_gene_node=Path(args.unique_gene_node),
        output_dir=Path(args.output_dir),
    )

    print(f"Wrote the initial drug-gene edges data to {output_paths[0]}")
    print(f"Wrote the final drug-gene edges data to {output_paths[1]}")
    print(f"Wrote the drug indications data to {output_paths[2]}")
    print(f"Wrote the drug SMILES data to {output_paths[3]}")
    print(f"Wrote the drug feature vectors to {output_paths[4]}")


if __name__ == "__main__":
    main()
