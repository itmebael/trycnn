"""
Migration script to migrate existing JSON data to Supabase
Run this script once to migrate your existing users.json and detection_results.json to Supabase
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("=" * 60)
    print("Supabase Migration Script")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ ERROR: .env file not found!")
        print()
        print("Please create a .env file with your Supabase credentials:")
        print("  SUPABASE_URL=https://your-project-id.supabase.co")
        print("  SUPABASE_ANON_KEY=your-anon-key-here")
        print()
        print("You can copy config_template.env to .env and fill in your credentials.")
        sys.exit(1)
    
    # Check if credentials are set
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
    
    if not supabase_url or not supabase_key or supabase_url.startswith("https://your-project") or supabase_key.startswith("your-anon"):
        print("❌ ERROR: Supabase credentials not configured!")
        print()
        print("Please update your .env file with valid Supabase credentials:")
        print("  SUPABASE_URL=https://your-project-id.supabase.co")
        print("  SUPABASE_ANON_KEY=your-anon-key-here")
        print()
        print("Get these from your Supabase project dashboard:")
        print("  Project Settings > API > Project URL and anon/public key")
        sys.exit(1)
    
    print("✓ Supabase credentials found")
    print()
    
    # Import migration function
    try:
        from db import migrate_json_to_supabase
    except ImportError as e:
        print(f"❌ ERROR: Could not import db module: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    
    # Check if files exist
    users_file = "users.json"
    results_file = "uploads/detection_results.json"
    
    users_exist = os.path.exists(users_file)
    results_exist = os.path.exists(results_file)
    
    if not users_exist and not results_exist:
        print("⚠️  No JSON files found to migrate.")
        print(f"   users.json: {'✓ exists' if users_exist else '✗ not found'}")
        print(f"   detection_results.json: {'✓ exists' if results_exist else '✗ not found'}")
        print()
        print("If you've already migrated or don't have existing data, you can skip this step.")
        sys.exit(0)
    
    print("Files to migrate:")
    if users_exist:
        print(f"  ✓ {users_file}")
    if results_exist:
        print(f"  ✓ {results_file}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with migration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Migration cancelled.")
        sys.exit(0)
    
    print()
    print("Starting migration...")
    print("-" * 60)
    
    try:
        migrated_users, migrated_results = migrate_json_to_supabase(users_file, results_file)
        print("-" * 60)
        print()
        print("✅ Migration completed successfully!")
        print(f"   Migrated {migrated_users} users")
        print(f"   Migrated {migrated_results} detection results")
        print()
        print("Note: Original JSON files are preserved. You can delete them manually")
        print("      after verifying the data in Supabase is correct.")
        
    except Exception as e:
        print()
        print("❌ ERROR during migration:")
        print(f"   {str(e)}")
        print()
        print("Please check:")
        print("  1. Your Supabase credentials are correct")
        print("  2. The database schema has been created (run supabase_schema.sql)")
        print("  3. Your internet connection is working")
        sys.exit(1)


if __name__ == "__main__":
    main()

