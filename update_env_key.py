"""
Quick script to update your Supabase anon key in .env file
Usage: python update_env_key.py your-anon-key-here
"""

import sys
import os

def update_anon_key(anon_key):
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print(f"❌ Error: {env_file} file not found!")
        print("   Make sure you're in the project root directory.")
        return False
    
    # Read current .env file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update the anon key line
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith('SUPABASE_ANON_KEY='):
            new_lines.append(f'SUPABASE_ANON_KEY={anon_key}\n')
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        print("❌ Error: Could not find SUPABASE_ANON_KEY line in .env file")
        return False
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Successfully updated SUPABASE_ANON_KEY in .env file!")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python update_env_key.py <your-anon-key>")
        print()
        print("Example:")
        print('  python update_env_key.py "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."')
        print()
        print("To get your anon key:")
        print("  1. Go to https://supabase.com/dashboard")
        print("  2. Select your project")
        print("  3. Settings → API → Copy 'anon public' key")
        sys.exit(1)
    
    anon_key = sys.argv[1]
    if update_anon_key(anon_key):
        print()
        print("✅ Your .env file is now configured!")
        print("   Next steps:")
        print("   1. Run the SQL schema in Supabase (supabase_schema.sql)")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Start your app: python app.py")
    else:
        sys.exit(1)

