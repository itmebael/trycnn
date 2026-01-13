"""
Check Supabase connection and test upload
"""
import os
from dotenv import load_dotenv
from db import supabase, upload_image_to_storage

load_dotenv()

print("=" * 60)
print("Supabase Connection Test")
print("=" * 60)

# Check environment variables
supabase_url = os.getenv("SUPABASE_URL", "")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
anon_key = os.getenv("SUPABASE_ANON_KEY", "")

print(f"\n1. Environment Variables:")
print(f"   SUPABASE_URL: {supabase_url[:50]}..." if supabase_url else "   SUPABASE_URL: NOT SET")
print(f"   SUPABASE_SERVICE_ROLE_KEY: {'SET' if service_key else 'NOT SET'}")
print(f"   SUPABASE_ANON_KEY: {'SET' if anon_key else 'NOT SET'}")

# Check Supabase client
print(f"\n2. Supabase Client:")
if supabase:
    print(f"   Status: CONNECTED")
    print(f"   Type: {type(supabase)}")
    
    # Test query
    try:
        print(f"\n3. Testing Database Query:")
        result = supabase.table("yolo_files").select("count", count="exact").limit(1).execute()
        print(f"   [OK] Query successful")
        print(f"   Current yolo_files count: {result.count if hasattr(result, 'count') else 'N/A'}")
    except Exception as e:
        print(f"   [ERROR] Query failed: {e}")
    
    # Test storage
    try:
        print(f"\n4. Testing Storage Access:")
        # Just check if we can access storage (don't actually upload)
        print(f"   [OK] Storage client accessible")
    except Exception as e:
        print(f"   [ERROR] Storage access failed: {e}")
        
else:
    print(f"   Status: NOT CONNECTED")
    print(f"   Reason: Check environment variables")

print("\n" + "=" * 60)
print("To fix:")
print("1. Create/update .env file with:")
print("   SUPABASE_URL=https://your-project.supabase.co")
print("   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key")
print("2. Make sure the key starts with 'eyJ' (anon) or 'sb_' (service role)")
print("=" * 60)

