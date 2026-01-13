# Windows Setup Guide for Supabase

## Quick Setup Steps

### Step 1: Create .env file (Already Done! ✓)
The `.env` file has been created from the template.

### Step 2: Edit .env File with Your Supabase Credentials

**On Windows, you can edit the `.env` file using:**

**Option A: Notepad (Easiest)**
```powershell
notepad .env
```

**Option B: VS Code (if installed)**
```powershell
code .env
```

**Option C: Any text editor**
Just open `.env` file with any text editor

### Step 3: Get Your Supabase Credentials

1. Go to your Supabase project dashboard: https://supabase.com/dashboard
2. Select your project
3. Click **Settings** (gear icon) in the left sidebar
4. Click **API** under Project Settings
5. Copy these values:
   - **Project URL**: Found under "Project URL" section
     - Example: `https://zqkqmjlepigpwfykwzey.supabase.co`
   - **anon public key**: Found under "Project API keys" → "anon public"
     - This is a long string starting with `eyJ...`

### Step 4: Update .env File

Edit the `.env` file and replace:

```env
SUPABASE_URL=https://zqkqmjlepigpwfykwzey.supabase.co
SUPABASE_ANON_KEY=paste-your-anon-key-here
FLASK_SECRET_KEY=generate-a-random-secret-key-here
```

**Important Notes:**
- Replace `https://zqkqmjlepigpwfykwzey.supabase.co` with your actual Project URL
- Replace `paste-your-anon-key-here` with your actual anon public key (the long string)
- Replace `generate-a-random-secret-key-here` with any random string (e.g., `my-secret-key-12345`)

### Step 5: Create Database Tables

1. Go to your Supabase dashboard
2. Click **SQL Editor** in the left sidebar
3. Click **New Query**
4. Open the file `supabase_schema.sql` from this project folder
5. Copy ALL the SQL code from that file
6. Paste it into the SQL Editor
7. Click **Run** (or press F5)
8. You should see "Success. No rows returned" - this means it worked!

### Step 6: Install Python Dependencies

Open PowerShell or Command Prompt in this folder and run:

```powershell
pip install -r requirements.txt
```

### Step 7: (Optional) Migrate Existing Data

If you have existing data in `users.json` or `uploads/detection_results.json`:

```powershell
python migrate_to_supabase.py
```

### Step 8: Start Your App

```powershell
python app.py
```

## Common Windows Commands Reference

| Task | Command |
|------|---------|
| Copy file | `copy source.txt dest.txt` |
| Create file | `notepad filename.txt` |
| View file | `type filename.txt` |
| List files | `dir` or `ls` (PowerShell) |
| Edit file | `notepad filename.txt` |

## Troubleshooting

### "Missing Supabase credentials" Error
- Make sure `.env` file exists (not `config_template.env`)
- Verify you edited `.env` and not just looked at it
- Restart your Flask app after editing `.env`
- Make sure there are NO spaces around the `=` sign

### "Table does not exist" Error
- Make sure you ran `supabase_schema.sql` in Supabase SQL Editor
- Check that all tables were created in Supabase Dashboard → Table Editor

### "pip is not recognized"
- Make sure Python is installed and added to PATH
- Try `python -m pip install -r requirements.txt` instead

### File Not Found Errors
- Make sure you're in the correct directory: `C:\xampp\htdocs\trycnn`
- Use `cd C:\xampp\htdocs\trycnn` to navigate there

## Verify Your Setup

After setting up, verify everything works:

1. ✅ Check `.env` file has your Supabase URL and key
2. ✅ Check Supabase dashboard → Table Editor shows `users` and `detection_results` tables
3. ✅ Run `python app.py` - should start without errors
4. ✅ Try registering a new user - should save to Supabase
5. ✅ Upload an image - should save detection result to Supabase

## Need Help?

- Check `SUPABASE_SETUP.md` for detailed setup guide
- Check `QUICK_START_SUPABASE.md` for quick reference
- Supabase Docs: https://supabase.com/docs

