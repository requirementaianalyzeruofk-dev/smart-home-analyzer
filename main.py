from db_manager import create_database, get_requirements
import sqlite3

# ----------------------------------------------------
# Main program starts here
# ----------------------------------------------------

def main():
    # Step 1: Create database and tables if they don't exist
    create_database()

    # Step 2: Fetch all requirements from database
    try:
        requirements = get_requirements()

        if not requirements:
            print("⚠️ No requirements found in the database.")
            print("Make sure you ran 'seed_requirements.py' first to insert data.")
            return

        # Step 3: Display each requirement nicely
        for req in requirements:
            print(f"- ID {req['id']}: {req['text']}")
            print(f"  Functional: {req['functional']}, Non-Functional: {req['non_functional']}")
            print(f"  Ambiguous: {req['ambiguous']}, Measurable: {req['measurable']}")
            print(f"  Notes: {req['notes']}")
            print(f"   File: {req['file']}")
            print("-" * 60)

    except sqlite3.Error as e:
        print("❌ Database error:", e)
    except Exception as ex:
        print("❌ Unexpected error:", ex)


# ----------------------------------------------------
# Run when executed directly
# ----------------------------------------------------
if __name__ == "__main__":
    main()