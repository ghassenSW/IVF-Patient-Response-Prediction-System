import pdfplumber
import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime


def anonymize_patient_ids(df):
    """
    Convert patient names to anonymized IDs in format 25XXX
    """
    unique_patients = df['patient_id'].unique()
    id_mapping = {}
    
    for idx, patient_name in enumerate(unique_patients, start=1):
        if pd.notna(patient_name):
            anonymized_id = f"25{idx:03d}"
            id_mapping[patient_name] = anonymized_id
    
    df['patient_id'] = df['patient_id'].map(lambda x: id_mapping.get(x, x) if pd.notna(x) else x)
    
    return df, id_mapping


def normalize_protocols(df):
    """
    Standardize protocol names to: "fixed antagonist", "flexible antagonist", "agonist"
    """
    protocol_mapping = {
        'flex antago': 'flexible antagonist',
        'flex antag': 'flexible antagonist',
        'flex anta': 'flexible antagonist',
        'flexible antagonist': 'flexible antagonist',
        'fix antag': 'fixed antagonist',
        'fixed anta': 'fixed antagonist',
        'fixed antagonist': 'fixed antagonist',
        'agoni': 'agonist',
        'agonist': 'agonist'
    }
    
    df['Protocol'] = df['Protocol'].str.lower().map(protocol_mapping)
    
    return df


def extract_pdf_data(pdf_path):
    all_text = []
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
            
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)
    
    full_text = "\n".join(all_text)
    
    patient_data = {
        'patient_id': None,
        'cycle_number': None,
        'Age': None,
        'Protocol': None,
        'AMH': None,
        'n_Follicles': None,
        'E2_day5': None,
        'AFC': None,
        'Patient Response': None
    }
    
    # Patient name
    name_match = re.search(r'Name\s*:\s*([A-Za-z]+\s+[A-Za-z])', full_text, re.IGNORECASE)
    if name_match:
        patient_data['patient_id'] = name_match.group(1).strip()
    
    # Age from birth date
    birth_match = re.search(r'Birth\s+date\s*:\s*(\d{2})/(\d{2})/(\d{2})', full_text, re.IGNORECASE)
    if birth_match:
        day, month, year = birth_match.groups()
        year_full = int('19' + year) if int(year) > 50 else int('20' + year)
        birth_date = datetime(year_full, int(month), int(day))
        age = (datetime.now() - birth_date).days / 365.25
        patient_data['Age'] = int(age)
    
    # AMH level
    amh_match = re.search(r'AMH\s*:\s*(\d+\.?\d*)', full_text, re.IGNORECASE)
    if amh_match:
        patient_data['AMH'] = float(amh_match.group(1))
    
    # Protocol
    protocol_match = re.search(r'Protocol\s*:\s*([^\n]+)', full_text, re.IGNORECASE)
    if protocol_match:
        patient_data['Protocol'] = protocol_match.group(1).strip().lower()
    
    # Cycle number
    cycle_match = re.search(r'Cycle\s+number\s*:\s*(\d+)', full_text, re.IGNORECASE)
    if cycle_match:
        patient_data['cycle_number'] = int(cycle_match.group(1))
    
    # Follicle count
    follicle_match = re.search(r'Number\s+Of\s+follicles\s*[=:]\s*(\d+)', full_text, re.IGNORECASE)
    if follicle_match:
        patient_data['n_Follicles'] = float(follicle_match.group(1))
    
    # E2 on day 5
    e2_day5_match = re.search(r'6/10\s+//\s+(\d+)', full_text)
    if e2_day5_match:
        patient_data['E2_day5'] = float(e2_day5_match.group(1))
    
    # AFC
    afc_match = re.search(r'AFC\s*:\s*(\d+\.?\d*)', full_text, re.IGNORECASE)
    if afc_match:
        patient_data['AFC'] = float(afc_match.group(1))
    
    # Response classification
    response_match = re.search(r'(?:patient\s+has\s+an?\s+|response\s*:\s*)(optimal|high|low)[-\s]?response', full_text, re.IGNORECASE)
    if response_match:
        patient_data['Patient Response'] = response_match.group(1).lower()
    
    return {
        'extracted_data': patient_data,
        'full_text': full_text,
        'tables': tables
    }


def add_to_csv(patient_data, csv_path):
    df = pd.read_csv(csv_path)
    new_row = pd.DataFrame([patient_data])
    df_updated = pd.concat([df, new_row], ignore_index=True)
    
    # Normalize protocols after adding new row
    df_updated = normalize_protocols(df_updated)
    
    df_updated.to_csv(csv_path, index=False)
    print(f"Added patient record. Total rows: {len(df_updated)}")
    return df_updated


def clean_dataset(input_path, output_path):
    """
    Clean raw dataset:
    1. Load data
    2. Anonymize patient IDs
    3. Normalize protocol names
    4. Save processed data
    """
    print(f"Loading data from {input_path.name}...")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    # Anonymize patient IDs
    print("\nAnonymizing patient IDs...")
    df, id_mapping = anonymize_patient_ids(df)
    
    # Normalize protocols
    print("Normalizing protocol names...")
    df = normalize_protocols(df)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path.name}")
    print(f"Total rows: {len(df)}")
    
    # Save ID mapping
    mapping_path = output_path.parent / "patient_id_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"ID mapping saved to {mapping_path.name}")
    
    return df


def main():
    project_root = Path(__file__).parent.parent.parent
    
    pdf_path = project_root / "data" / "raw" / "sample.pdf"
    raw_csv_path = project_root / "data" / "raw" / "patients.csv"
    clean_csv_path = project_root / "data" / "processed" / "patients_clean.csv"
    text_output_path = project_root / "data" / "processed" / "extracted_pdf_text.txt"
    
    # Extract PDF data and add to raw CSV
    print(f"Extracting data from {pdf_path.name}...")
    result = extract_pdf_data(pdf_path)
    patient_data = result['extracted_data']
    
    # Save extracted text
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write("PDF TEXT:\n\n")
        f.write(result['full_text'])
        f.write("\n\nEXTRACTED DATA:\n\n")
        f.write(json.dumps(patient_data, indent=2))
    
    add_to_csv(patient_data, raw_csv_path)
    print("Extraction complete!")
    
    # Clean the dataset
    print("\n" + "="*50)
    print("Starting dataset cleaning...")
    print("="*50 + "\n")
    df_clean = clean_dataset(raw_csv_path, clean_csv_path)
    
    print("\nSample of cleaned data:")
    print(df_clean.head(10))
    
    print("\nProtocol distribution:")
    print(df_clean['Protocol'].value_counts())
    
    return patient_data


if __name__ == "__main__":
    main()
