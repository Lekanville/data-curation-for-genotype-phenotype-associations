import time
import requests
import argparse
import mygene
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from collections import Counter


# Fetch Features using the mapped Entrez IDs
def get_gene_features_with_entrez(entrez_ids):
    """
    Gets the gene features using the Entrez IDs.
    """
    mg = mygene.MyGeneInfo()
    
    # Query MyGene.info
    # We ask for the 'ensembl.gene' field
    gene_data = mg.querymany(entrez_ids,
                            fields="symbol,exac,go,genomic_pos,uniprot", 
                            species='human')

    # Filter out 'notfound' records
    filtered_gene_data = [
        record for record in gene_data 
        if 'notfound' not in record
    ]
            
    # return pd.DataFrame(filtered_gene_data)
    return filtered_gene_data

# for pLi and Gene length
def get_gene_pl_gl_with_entrez(gene_features_dict):
    # 1. Initialize the final feature DataFrame
    entrez_and_symbol = [
        {'Entrez_ID':gene_features_dict[i]['_id'], 'Gene_Symbol':gene_features_dict[i]['symbol']} for i,j in enumerate(gene_features_dict)
    ]
    df_processed_features = pd.DataFrame(entrez_and_symbol)
    
    # 2. Extract pLI Score and Gene Length by iterating and safely accessing nested fields
    
    pli_scores = []
    gene_lengths = []
    pli_imputed_flags = []
    gene_lengths_imputed_flags = []
    
    for record in gene_features_dict:
        p_li = 0.0
        pli_imputed = True # Start by assuming data is missing
        exac_data = record.get('exac') 
    
        p_li_raw = None  # Use None to track if we actually found a value
        if exac_data and isinstance(exac_data, dict):
            # Try the nested 'all' dict first
            if 'all' in exac_data and isinstance(exac_data.get('all'), dict):
                p_li_raw = exac_data['all'].get('p_li') # Defaults to None if key missing
            
            # If still None, try the top level of exac_data
            if p_li_raw is None:
                p_li_raw = exac_data.get('p_li')
        
        # Final check to set score and flag
        if p_li_raw is not None:
            # We found a real number (could be 0.0, 0.5, 1.0, etc.)
            p_li = float(p_li_raw)
            pli_imputed = False
        else:
            # Data was truly missing from the record
            p_li = 0.0
            pli_imputed = True
    
        pli_scores.append(p_li)
        pli_imputed_flags.append(pli_imputed)
    
        # --- Gene Length (Genomic Size) ---
        size = np.nan
        size_imputed = True # Start by assuming data is missing
        genomic_pos_data = record.get('genomic_pos')
        
        # genomic_pos can be a single dict or a list of dicts (for different assemblies/transcripts)
        if isinstance(genomic_pos_data, list):
            # For simplicity, calculate length from the start/end of the first record
            pos_data = genomic_pos_data[0] 
        elif isinstance(genomic_pos_data, dict):
            pos_data = genomic_pos_data
        else:
            pos_data = {}
    
        if pos_data and pos_data.get('end') and pos_data.get('start'):
            size = abs(pos_data['end'] - pos_data['start']) + 1
            size_imputed = False # We were able to calculate the gene length, so we set to false
        
        gene_lengths.append(size)
        gene_lengths_imputed_flags.append(size_imputed)
    
    
    # 3. Add the new features to the processed DataFrame
    df_processed_features['pLI_Score'] = pli_scores
    df_processed_features['pLI_Imputed_Flag'] = pli_imputed_flags
    df_processed_features['Gene_Length'] = gene_lengths
    df_processed_features['Gene_Length_Imputted_Flag'] = gene_lengths_imputed_flags
    
    # 4. Handle NaN/missing values for Gene_Length
    # Impute missing lengths with the median of the found lengths
    median_length = df_processed_features['Gene_Length'].median()
    # Ensure median_length is not NaN itself before using fillna
    median_length_safe = median_length if not np.isnan(median_length) else 1000 # Default to 1kb if no lengths found as median length
    
    df_processed_features['Gene_Length'] = df_processed_features['Gene_Length'].fillna(median_length_safe)
    
    print("Successfully extracted pLI Score and Gene Length.")
    return (df_processed_features)
    
def get_gene_mw(gene_features_dict, df_processed_features):
    # List to hold the extracted molecular weights
    
    # UniProt API URL
    UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/search"
    
    mw_map = {}
    
    # Dictionary to hold {UniProt_Accession: Entrez_ID}
    accession_to_entrez = {} 
    
    # Create a list of primary accessions to query
    accessions_to_query = [] 
    
    for record in gene_features_dict:
        entrez_id = str(record.get('_id'))
        uniprot_data = record.get('uniprot')
        
        if uniprot_data and isinstance(uniprot_data, dict):
            # Look for the primary Swiss-Prot ID
            swissprot_id = uniprot_data.get('Swiss-Prot')
            
            if isinstance(swissprot_id, list):
                if swissprot_id:
                    # Pick the first accession in the list
                    primary_accession = swissprot_id[0] 
                else:
                    continue # Skip if the list is empty
            elif isinstance(swissprot_id, str):
                primary_accession = swissprot_id
            else:
                continue # Skip if the data type is neither string nor list (e.g., None or NaN)
        
            # Now primary_accession is guaranteed to be a hashable string
            if primary_accession and primary_accession not in accessions_to_query:
                accessions_to_query.append(primary_accession)
                # Store the map {primary_accession: entrez_id}
                accession_to_entrez[primary_accession] = entrez_id
    
    print(f"Found {len(accessions_to_query)} UniProt accessions to query.")
    
    
    # --- Step 2: Batch Query UniProt API for Molecular Weight ---
    
    BATCH_SIZE = 100 
    fields = "id,mass" # Request the ID and the molecular mass
    
    print("Starting UniProt API query for Molecular Weight...")
    
    mw_g_Imputed_Flag = True
    
    for i in range(0, len(accessions_to_query), BATCH_SIZE):
        batch = accessions_to_query[i:i + BATCH_SIZE]
        query_string = " OR ".join([f'accession:{acc}' for acc in batch])
        
        params = {
            'query': query_string,
            'format': 'json',
            'fields': fields,
            'size': BATCH_SIZE 
        }
        
        try:
            response_uniprot = requests.get(UNIPROT_API_URL, params=params)
            response_uniprot.raise_for_status()
            data = response_uniprot.json()
            
            for result in data.get('results', []):
                accession = result.get('primaryAccession')
                # Molecular weight is found in the 'sequence' object under 'molWeight'
                mw_g = result.get('sequence', {}).get('molWeight')
                
                if accession and mw_g:
                    entrez_id = accession_to_entrez.get(accession)
                    if entrez_id:
                        # UniProt returns mass in Daltons
                        mw_map[entrez_id] = float(mw_g)  
    
            time.sleep(0.5) 
    
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during UniProt query: {e}")
            time.sleep(5) 
    
    print("✅ Molecular Weight data retrieval complete.")
    
    
    # --- Step 3: Map MW to DataFrame and Impute Missing ---
    
    # Ensure the Entrez ID column is string type for mapping
    df_processed_features['Entrez_ID'] = df_processed_features['Entrez_ID'].astype(str)
    
    # Apply the map (keys are Entrez IDs, values are MW)
    df_processed_features['Molecular_Weight'] = df_processed_features['Entrez_ID'].map(mw_map)
    
    # Calculate the median MW from successfully retrieved values
    median_mw = df_processed_features['Molecular_Weight'].median()
    # Fallback value (approx 50kDa is a safe biological average if everything fails)
    median_mw_safe = median_mw if not np.isnan(median_mw) else 50000.0
    
    # Create the binary flag BEFORE filling the NaNs
    df_processed_features["M_W_Imputed_Flag"] = df_processed_features['Molecular_Weight'].isna().astype(int)
    
    # Fill with the safe median
    df_processed_features['Molecular_Weight'] = df_processed_features['Molecular_Weight'].fillna(median_mw_safe)
    
    # Impute NaN values with the median MW and ensure type is float
    df_processed_features['Molecular_Weight'] = df_processed_features['Molecular_Weight'].astype(float)
    
    print(f"✅ Molecular Weight feature successfully retrieved from UniProt and added.")
    print(f"Median Molecular Weight used for imputation: {median_mw:.2f} Daltons")
    # print(df_gene_features_final[['Gene_Symbol', 'pLI_Score', 'PPI_Count', 'Molecular_Weight']].head())
    return (df_processed_features)


def get_go_vocab(gene_features_dict, vocab_size):
    # Initialize lists to hold all terms by category
    bp_terms_all = [] # Biological Process
    mf_terms_all = [] # Molecular Function
    cc_terms_all = [] # Cellular Component
    
    for record in gene_features_dict:
        go_data = record.get('go')
        
        if go_data and isinstance(go_data, dict):
            # Biological Process (BP)
            bp_list = go_data.get('BP', [])
            if isinstance(bp_list, list):
                bp_terms_all.extend([item['term'].lower() for item in bp_list])
                
            # Molecular Function (MF)
            mf_list = go_data.get('MF', [])
            if isinstance(mf_list, list):
                mf_terms_all.extend([item['term'].lower() for item in mf_list])
                
            # Cellular Component (CC)
            cc_list = go_data.get('CC', [])
            if isinstance(cc_list, list):
                cc_terms_all.extend([item['term'].lower() for item in cc_list])
        # Note: Genes with NaN in 'go' are skipped (e.g., MIR genes), which is correct.
    
    print("✅ GO terms flattened and categorized.")


    # Create frequency counters
    bp_counter = Counter(bp_terms_all)
    mf_counter = Counter(mf_terms_all)
    cc_counter = Counter(cc_terms_all)
    
    # Select the top N terms for the vocabulary
    bp_vocab = [term for term, count in bp_counter.most_common(vocab_size)]
    mf_vocab = [term for term, count in mf_counter.most_common(vocab_size)]
    cc_vocab = [term for term, count in cc_counter.most_common(vocab_size)]
    
    print("-" * 50)
    print(f"Total unique GO terms found: {len(bp_counter) + len(mf_counter) + len(cc_counter)}")
    print(f"BP Vocabulary Size (Top {vocab_size}): {len(bp_vocab)}")
    print(f"MF Vocabulary Size (Top {vocab_size}): {len(mf_vocab)}")
    print(f"CC Vocabulary Size (Top {vocab_size}): {len(cc_vocab)}")
    print("-" * 50)
    print(f"Example BP Terms: {bp_vocab[:5]}")
    print(f"Example MF Terms: {mf_vocab[:5]}")
    print(f"Example CC Terms: {cc_vocab[:5]}")

    return (bp_vocab, mf_vocab, cc_vocab)


def encode_go_terms(gene_record, vocab, category):
    """Generates a multi-hot vector for a single gene's GO terms."""
    vector = np.zeros(len(vocab))
    
    go_data = gene_record.get('go')
    if go_data and isinstance(go_data, dict):
        # Get the list of terms for the specific category (BP, MF, or CC)
        category_list = go_data.get(category, [])
        
        if isinstance(category_list, list):
            # Convert the gene's terms to a set for fast lookup
            gene_term_set = {item['term'].lower() for item in category_list}
            
            # Populate the vector
            for i, term in enumerate(vocab):
                if term in gene_term_set:
                    vector[i] = 1
    return vector

# --- Improved GO Encoding with Evidence Flag ---
def encode_go_terms_with_metadata(gene_record, bp_vocab, mf_vocab, cc_vocab):
    """Generates vectors and a 'completeness' score for GO annotations."""
    # Check if ANY GO data exists
    go_data = gene_record.get('go', {})
    
    # If go_data is totally missing or not a dict, it's clearly missing
    if not isinstance(go_data, dict) or not go_data:
        return np.zeros(len(bp_vocab)), np.zeros(len(mf_vocab)), np.zeros(len(cc_vocab)), 1 # 1 = Imputed/Missing
    
    # helper to check if categories have actual entries
    has_bp = len(go_data.get('BP', [])) > 0
    has_mf = len(go_data.get('MF', [])) > 0
    has_cc = len(go_data.get('CC', [])) > 0
    
    # Functional_Imputed_Flag is 0 if we found at least one annotation in any category
    imputed_flag = 0 if (has_bp or has_mf or has_cc) else 1
    
    # Use your existing logic to get the actual vectors
    bp_v = encode_go_terms(gene_record, bp_vocab, 'BP')
    mf_v = encode_go_terms(gene_record, mf_vocab, 'MF')
    cc_v = encode_go_terms(gene_record, cc_vocab, 'CC')
    
    return bp_v, mf_v, cc_v, imputed_flag


#For Gene expression and PPI
def gene_symbols_to_string_id(df_with_gene_symbols):
    # Create a map {Symbol: Entrez ID} from the final DataFrame
    symbol_to_entrez_map = df_with_gene_symbols.set_index('Gene_Symbol')['Entrez_ID'].astype(str).to_dict()
    
    # Use the Gene Symbols for the initial query, as they are non-ambiguous text
    gene_symbols_to_query = df_with_gene_symbols['Gene_Symbol'].dropna().unique().tolist()
    
    STRING_MAP_URL = "https://string-db.org/api/tsv/get_string_ids?"
    string_id_to_symbol_map = {} 
    
    # ... (API query loop using gene_symbols_to_query) ...
    BATCH_SIZE = 200 
    
    print("Mapping Gene Symbols to STRING IDs...")
    
    for i in range(0, len(gene_symbols_to_query), BATCH_SIZE):
        # ... (Batch and params setup using identifiers=gene_symbols_to_query) ...
        batch = gene_symbols_to_query[i:i + BATCH_SIZE]
        identifiers = "%0d".join(batch) 
        
        params = {
            "identifiers": identifiers,
            "species": 9606,
            "caller_identity": "your_project_name"
        }
        
        try:
            response_string_map = requests.get(STRING_MAP_URL, params=params)
            response_string_map.raise_for_status()
            
            lines = response_string_map.text.strip().split('\n')
            if len(lines) > 1:
                header = lines[0].split('\t')
                # Columns to rely on: preferredName and stringId
                pref_name_idx = header.index('preferredName')
                string_id_idx = header.index('stringId')
                
                for line in lines[1:]: 
                    parts = line.split('\t')
                    if len(parts) > max(pref_name_idx, string_id_idx):
                        symbol_used = parts[pref_name_idx] # This is the Gene Symbol
                        string_id = parts[string_id_idx]
                        
                        # Store the map {STRING ID: Gene Symbol}
                        string_id_to_symbol_map[string_id] = symbol_used 
            
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during STRING MAP query: {e}")
            time.sleep(5)
    
    official_string_ids = list(string_id_to_symbol_map.keys())
    print(f"Mapped {len(official_string_ids)} Symbols to STRING IDs.")

    return (string_id_to_symbol_map, official_string_ids, symbol_to_entrez_map)


def get_ppi_counts(string_id_to_symbol_map, official_string_ids, symbol_to_entrez_map, df_current):
    # 3. Query partners using the official STRING IDs (same logic as before)
    STRING_API_URL = "https://string-db.org/api/tsv/interaction_partners?"
    ppi_counts_by_string_id = Counter()

    BATCH_SIZE = 200 
    
    print("2. Querying interaction partners using official STRING IDs...")
    
    # ... (Interaction query loop using official_string_ids to populate ppi_counts_by_string_id) ...
    for i in range(0, len(official_string_ids), BATCH_SIZE):
        # ... (Batch and params setup using identifiers=official_string_ids) ...
        batch = official_string_ids[i:i + BATCH_SIZE]
        identifiers = "%0d".join(batch) 
        
        params = {
            "identifiers": identifiers,
            "species": 9606,
            "required_score": 700, 
            "limit": 1000,
        }
        
        try:
            response_string_ppi = requests.get(STRING_API_URL, params=params)
            response_string_ppi.raise_for_status()
            data = response_string_ppi.text.strip()
            
            if data:
                header = data.split('\n')[0].split('\t')
                string_id_a_idx = header.index('stringId_A') 
                
                for line in data.split('\n')[1:]:
                    row = line.split('\t')
                    if len(row) > string_id_a_idx:
                        string_id = row[string_id_a_idx]
                        ppi_counts_by_string_id[string_id] += 1
                    
            time.sleep(0.5) 
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during STRING PARTNER query: {e}")
            time.sleep(5) 
    
    
    # --- 4. Final Mapping (The Symbol-to-Entrez Bridge) ---
    
    # Create the final DataFrame for merging, keyed by Entrez ID
    final_ppi_data = []
    for string_id, count in ppi_counts_by_string_id.items():
        # 1. Get the Gene Symbol using the successful map
        symbol = string_id_to_symbol_map.get(string_id)
        if symbol:
            # 2. Get the Entrez ID using the Symbol (guaranteed match)
            entrez_id = symbol_to_entrez_map.get(symbol)
            if entrez_id:
                final_ppi_data.append({'Entrez_ID': entrez_id, 'PPI_Count_New': count})
    
    # --- 5. Use your successful merge logic ---
    df_ppi_counts = pd.DataFrame(final_ppi_data)
    
    # Ensure merge key types are consistent (String)
    df_ppi_counts['Entrez_ID'] = df_ppi_counts['Entrez_ID'].astype(str)
    df_current['Entrez_ID'] = df_current['Entrez_ID'].astype(str)
    
    # Perform the merge (left join)
    df_current = df_current.merge(
        df_ppi_counts, 
        on='Entrez_ID', 
        how='left' 
    )

    return df_current


# Gene Expression Specificity - HPA Specificity Score
def get_hpatss(df_current):
    HPA_API_BASE_URL = "https://www.proteinatlas.org/api/search_download.php"
    # CRITICAL FIX: Use the correct, short abbreviations for API request
    HPA_COLUMNS = "g,rnatss" 
    BATCH_SIZE = 1 
    SCORE_KEY = "RNA tissue specificity score" # This is the key in the returned JSON
    
    # Extract the list of unique Gene Symbols
    gene_symbols_to_query = df_current['Gene_Symbol'].dropna().unique().tolist()
    
    # Dictionary to store {Gene_Symbol: rnatss_score}
    specificity_map = {}
    
    print(f"Starting HPA API query for {len(gene_symbols_to_query)} gene symbols, one by one...")
    
    for i in range(0, len(gene_symbols_to_query)):
        # The batch size is 1, so we just take the single symbol
        symbol_to_query = gene_symbols_to_query[i]
        
        params = {
            # Query with the single gene symbol
            'search': symbol_to_query, 
            'columns': HPA_COLUMNS, # <--- CORRECTED: Using 'g,rnatss'
            'format': 'json', 
            'compress': 'no'
        }
        
        try:
            response_hpa = requests.get(HPA_API_BASE_URL, params=params)
            response_hpa.raise_for_status()
            
            data = response_hpa.json()
            
            if isinstance(data, list) and data:
                entry = data[0]
                symbol = entry.get('Gene') 
                # We use the full string key to retrieve the value from the JSON
                score_str = entry.get(SCORE_KEY) 
                
                if symbol and score_str:
                     specificity_map[symbol] = float(score_str)
            
            time.sleep(0.3) 
    
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during HPA query for symbol {symbol_to_query}: {e}")
            time.sleep(5) 
        except ValueError:
            print(f"❌ Error decoding JSON response for symbol {symbol_to_query}.")
            time.sleep(0.3)
            
    print(f"✅ HPA Specificity Score data retrieval complete. Mapped {len(specificity_map)} symbols out of {len(gene_symbols_to_query)}.")
    
    # --- Final Mapping and Imputation ---
    df_current['Expression_Specificity_Score'] = df_current['Gene_Symbol'].map(specificity_map)
    
    #Imputation for missing Expression Specificity Score
    df_current["Ex_Sp_Flag_Imputted"] = df_current['Expression_Specificity_Score'].isna().astype(int)
    median_score = df_current['Expression_Specificity_Score'].median()
    median_score_safe = median_score if not np.isnan(median_score) else 1.0
    
    df_current['Expression_Specificity_Score'] = df_current['Expression_Specificity_Score'].fillna(median_score_safe)
    df_current['Expression_Specificity_Score'] = df_current['Expression_Specificity_Score'].astype(float)
    
    
    print(f"All Gene Node Features (X_G) finalized.")
    print(f"Median Specificity Score used for imputation: {median_score:.4f}")
    print(df_current[['Entrez_ID', 'Gene_Symbol', 'Expression_Specificity_Score', "Ex_Sp_Flag_Imputted"]].head())

    return (df_current)


def get_gene_features(
    unique_gene_list: Path, 
    vocab_size: int, 
    output_dir: Path
    ) -> Path:
    df_elite_genes = pd.read_csv(unique_gene_list)
    df_elite_genes['Entrez_ID'] = df_elite_genes['Entrez_ID'].apply(lambda x: str(x).split('.')[0])

    entrez_gene_list = df_elite_genes['Entrez_ID'].unique()
    entrez_gene_list_cleaned = [i for i in entrez_gene_list if i != 'nan']
    print(f"Number of unique Entrez IDs: {len(entrez_gene_list_cleaned)}")

    # Fetch gene features using the cleaned Entrez IDs
    gene_features = get_gene_features_with_entrez(entrez_gene_list_cleaned)

    # 1 and 2. For pLi and Gene length
    df_gene_pl_gl = get_gene_pl_gl_with_entrez(gene_features)

    # 3. For Molecular weight
    df_gene_mw = get_gene_mw(gene_features, df_gene_pl_gl)

    # 4. For the Gene Ontology Terms
    bp_vocab, mf_vocab, cc_vocab = get_go_vocab(gene_features, vocab_size)

    # Apply this to your loop
    go_results = [encode_go_terms_with_metadata(r, bp_vocab, mf_vocab, cc_vocab) for r in gene_features]

    # Unpack results
    bp_vectors, mf_vectors, cc_vectors, functional_flags = zip(*go_results)

    # Convert vectors to DataFrames
    df_bp = pd.DataFrame(bp_vectors, columns=[f'GO_BP_{i}' for i in range(len(bp_vocab))])
    df_mf = pd.DataFrame(mf_vectors, columns=[f'GO_MF_{i}' for i in range(len(mf_vocab))])
    df_cc = pd.DataFrame(cc_vectors, columns=[f'GO_CC_{i}' for i in range(len(cc_vocab))])

    # Add the flag to the final gene dataframe
    df_gene_mw['GO_Data_Missing_Flag'] = list(functional_flags)

    # To ensure the index matches for concatenation
    df_gene_go = df_gene_mw.reset_index(drop=True)
    df_gene_go_final = pd.concat([df_gene_go, df_bp, df_mf, df_cc], axis=1)

    print("Final gene node features created...")
    print(f"Total number of features per Gene Node: {df_gene_go_final.shape[1] - 2}") # Subtract Entrez_ID and Gene_Symbol
    print(f"Final DataFrame shape: {df_gene_go_final.shape}")
    df_gene_go_final.head()

    #For Gene expression and PPI
    string_id_to_symbol_map, official_string_ids, symbol_to_entrez_map = gene_symbols_to_string_id(df_gene_go_final)

    # 5. For PPI counts
    df_ppi_counts = get_ppi_counts(string_id_to_symbol_map, official_string_ids, symbol_to_entrez_map, df_gene_go_final)


    # Create a set of Entrez IDs that have a confirmed STRING ID
    # (Using symbol_to_entrez_map and your string_id_to_symbol_map)
    entrez_ids_with_string_data = set()

    for string_id, symbol in string_id_to_symbol_map.items():
        entrez_id = symbol_to_entrez_map.get(symbol)
        if entrez_id:
            entrez_ids_with_string_data.add(str(entrez_id))


    # Define the refined Flagging logic
    def set_ppi_flag(row):
        # If PPI_Count_New is > 0, we found interactions. Flag = 0.
        if row['PPI_Count_New'] > 0:
            return 0
        
        # If the gene ID is in our 'known' list, STRING confirmed 0 interactions.
        # This is a real biological 0. Flag = 0.
        if str(row['Entrez_ID']) in entrez_ids_with_string_data:
            return 0
        
        # Otherwise, the gene isn't in STRING at all. Flag = 1 (Imputed).
        return 1

    # Create the flag and fill the counts
    df_ppi_counts['PPI_Count_Imputed_Flag'] = df_ppi_counts.apply(set_ppi_flag, axis=1)
    df_ppi_counts['PPI_Count'] = df_ppi_counts['PPI_Count_New'].fillna(0).astype(int)

    # Drop the temporary column
    df_ppi_counts_with_flag = df_ppi_counts.drop(columns=['PPI_Count_New'])

    # Finalize the PPI_Count column
    # We fill NaNs (genes that had 0 partners or were unmappable) with 0.
    print(f"PPI Count acquisition complete and successfully merged via Symbol Bridge.")
    print(f"Non-zero PPI counts found: {sum(1 for count in df_ppi_counts_with_flag['PPI_Count'] if count > 0)}")
    print(df_ppi_counts_with_flag[['Entrez_ID', 'Gene_Symbol', 'pLI_Score', 'PPI_Count', 'PPI_Count_Imputed_Flag']].head())


    # 6. For Gene Expression Specificity - HPA Specificity Score
    df_hpatss = get_hpatss(df_ppi_counts_with_flag)

    # 7. The final dataset
    df_gene_node = df_hpatss.drop(["Gene_Symbol"], axis = 1)
    df_gene_node.set_index("Entrez_ID", inplace=True)

    # 7. Save the final DataFrame to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_features_all = output_dir / "gene_features_all.csv"
    gene_node_final = output_dir / "gene_node_final.csv"

    df_hpatss.to_csv(gene_features_all, index=False)
    df_gene_node.to_csv(gene_node_final, index=True)

    return (gene_features_all, gene_node_final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get gene features and imputation for missing data")
    parser.add_argument("--unique-gene-list", default="df_gene_collapsed.csv", help="Path to the df_gene_collapsed CSV")
    parser.add_argument("--vocab-size", default="75", help="Vocabulary size for the gene features")
    parser.add_argument("--output-dir", default="output/gene_features", help="Directory for the gene list outputs")
    args = parser.parse_args()

    output_paths = get_gene_features(
        unique_gene_list=Path(args.unique_gene_list),
        vocab_size=int(args.vocab_size),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote the complete gene features to {output_paths[0]} and gene node data to {output_paths[1]}")


if __name__ == "__main__":
    main()
