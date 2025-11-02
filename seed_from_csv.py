import csv
from db_manager import insert_requirement, create_database

# 1. Define file names
CSV_FILE = "requirements.csv"
FILE_NAME_IN_DB = "extended_csv_requirements" 

# 2. Create tables (Safe: won't recreate existing ones)
create_database() 

print(f"Reading requirements from {CSV_FILE}...")

# 3. Open CSV file and read data
with open(CSV_FILE, mode='r', encoding='utf-8') as file:
    # Use DictReader to treat the data like a dictionary (column name is the key)
    reader = csv.DictReader(file)
    
    # Loop through each row in the CSV file
    for row in reader:
        
        # 4. Convert 'label' text to a Boolean value (True if 'Functional', False otherwise)
        is_functional = (row['label'] == 'Functional')
        
        # 5. Insert the requirement into the database
        insert_requirement(
            # Get the requirement text from the 'requirement' column
            requirement_text=row['requirement'],
            is_functional=is_functional,
            # Assume non-ambiguous and measurable for the new data
            is_ambiguous=False, 
            is_measurable=True,
            notes=f"Source: {CSV_FILE}. Type: {row['label']}",
            file_name=FILE_NAME_IN_DB
        )
        print(f"✅ Inserted: {row['requirement']}")

print("✅ Data seeding from CSV completed successfully!")