from database import db

# Call the clear_database function
if __name__ == "__main__":
    print("Clearing all records from the database...")
    db.clear_database()
    print("Database has been cleared successfully!")