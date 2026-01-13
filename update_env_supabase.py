"""
Update .env file with Supabase Service Role Key
"""
import os
from pathlib import Path

# The service role key you provided
SERVICE_ROLE_KEY = "sb_secret_JGU_P__Kcf0yc0nB2m3aPA_S0upaaji"

# Read existing .env or create new
env_path = Path(".env")
env_content = {}

if env_path.exists():
    print("Reading existing .env file...")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_content[key.strip()] = value.strip()
else:
    print("Creating new .env file...")

# Update with service role key
env_content["SUPABASE_SERVICE_ROLE_KEY"] = SERVICE_ROLE_KEY

# Ensure other required keys exist
if "SUPABASE_URL" not in env_content:
    env_content["SUPABASE_URL"] = "https://zqkqmjlepigpwfykwzey.supabase.co"

if "SUPABASE_ANON_KEY" not in env_content:
    print("[WARN] SUPABASE_ANON_KEY not found. You may need to add it manually.")

# Write back to .env
print("\nUpdating .env file...")
with open(env_path, "w") as f:
    f.write("# Supabase Configuration\n")
    f.write(f"SUPABASE_URL={env_content.get('SUPABASE_URL', '')}\n")
    if "SUPABASE_ANON_KEY" in env_content:
        f.write(f"SUPABASE_ANON_KEY={env_content['SUPABASE_ANON_KEY']}\n")
    f.write(f"SUPABASE_SERVICE_ROLE_KEY={SERVICE_ROLE_KEY}\n")
    f.write("\n# Flask Configuration\n")
    if "FLASK_SECRET_KEY" in env_content:
        f.write(f"FLASK_SECRET_KEY={env_content['FLASK_SECRET_KEY']}\n")
    else:
        f.write("FLASK_SECRET_KEY=change-this-secret-key-to-something-random\n")
    if "FLASK_ENV" in env_content:
        f.write(f"FLASK_ENV={env_content['FLASK_ENV']}\n")
    else:
        f.write("FLASK_ENV=development\n")

print("[OK] .env file updated successfully!")
print(f"\nAdded SUPABASE_SERVICE_ROLE_KEY: {SERVICE_ROLE_KEY[:20]}...")
print("\nNote: The service role key you provided starts with 'sb_secret_'")
print("This might be a custom format. If uploads fail, check:")
print("1. The key is correct in Supabase Dashboard > Settings > API")
print("2. The key should have 'service_role' permissions")
print("3. Try using the full key from Supabase dashboard")

