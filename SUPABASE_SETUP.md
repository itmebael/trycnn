# Supabase Setup Guide

This guide will help you set up Supabase for your Pechay Detection System.

## Step 1: Create a Supabase Account and Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up for a free account (or log in if you already have one)
3. Click "New Project"
4. Fill in:
   - **Project Name**: pechay-detection (or any name you prefer)
   - **Database Password**: Create a strong password (save this!)
   - **Region**: Choose the closest region to you
5. Click "Create new project" and wait for it to initialize (2-3 minutes)

## Step 2: Get Your Supabase Credentials

1. In your Supabase project dashboard, go to **Settings** (gear icon) > **API**
2. Copy the following:
   - **Project URL**: Found under "Project URL" (e.g., `https://xxxxx.supabase.co`)
   - **anon public key**: Found under "Project API keys" > "anon public"

## Step 3: Create the Database Schema

1. In your Supabase dashboard, click on **SQL Editor** in the left sidebar
2. Click "New Query"
3. Open the `supabase_schema.sql` file from this project
4. Copy all the SQL code and paste it into the SQL Editor
5. Click "Run" (or press Ctrl+Enter)
6. You should see "Success. No rows returned" - this means the tables were created successfully!

## Step 4: Configure Your Application

1. Copy `config_template.env` to `.env`:
   ```bash
   cp config_template.env .env
   ```
   
   Or create a `.env` file manually with:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   FLASK_SECRET_KEY=change-this-secret-key-to-something-random
   FLASK_ENV=development
   ```

2. Edit `.env` and replace:
   - `SUPABASE_URL` with your Project URL from Step 2
   - `SUPABASE_ANON_KEY` with your anon public key from Step 2
   - `FLASK_SECRET_KEY` with a random secret string (you can generate one online)

## Step 5: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install the Supabase Python client and other dependencies.

## Step 6: Migrate Existing Data (Optional)

If you have existing data in `users.json` or `uploads/detection_results.json`, you can migrate it to Supabase:

```bash
python migrate_to_supabase.py
```

This script will:
- Check for existing JSON files
- Migrate users to the `users` table
- Migrate detection results to the `detection_results` table
- Preserve your original JSON files

## Step 7: Verify the Setup

1. Start your Flask application:
   ```bash
   python app.py
   ```

2. Try to:
   - Register a new user
   - Login with existing credentials
   - Upload an image and see if it's saved to Supabase

3. Check your Supabase dashboard:
   - Go to **Table Editor** in the left sidebar
   - You should see `users` and `detection_results` tables
   - Verify that data is being inserted when you use the app

## Troubleshooting

### Error: "Missing Supabase credentials"
- Make sure you created a `.env` file (not just `config_template.env`)
- Verify that `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set correctly
- Restart your Flask application after creating/modifying `.env`

### Error: "relation does not exist" or "table not found"
- Make sure you ran the SQL schema (`supabase_schema.sql`) in the SQL Editor
- Check that all tables were created successfully in the Table Editor

### Error: "permission denied" or "RLS policy violation"
- The schema includes Row Level Security (RLS) policies
- If you're having issues, you can temporarily disable RLS in Supabase:
  1. Go to **Table Editor**
  2. Select a table
  3. Click **Settings** > **Disable RLS** (for testing only!)
- The default policies allow public reads and inserts

### Migration script fails
- Ensure your Supabase credentials are correct
- Make sure the database schema has been created
- Check that your internet connection is working
- Verify the JSON files exist and are valid JSON

## Database Schema Overview

### `users` table
- `id` (UUID, Primary Key)
- `username` (VARCHAR, Unique)
- `email` (VARCHAR, Unique)
- `password` (VARCHAR, Hashed)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### `detection_results` table
- `id` (UUID, Primary Key)
- `filename` (VARCHAR)
- `condition` (VARCHAR: 'Healthy' or 'Diseased')
- `confidence` (DECIMAL)
- `timestamp` (TIMESTAMP)
- `image_path` (TEXT)
- `user_id` (UUID, Foreign Key to users)
- `all_probabilities` (JSONB)
- `recommendations` (JSONB)
- `created_at` (TIMESTAMP)

## Security Notes

- The `.env` file should **never** be committed to Git (it's in .gitignore)
- The `SUPABASE_ANON_KEY` is safe to use in client-side code (it's public)
- For admin operations, you might want to use `SUPABASE_SERVICE_ROLE_KEY`, but keep it secure!
- Row Level Security (RLS) is enabled by default - adjust policies as needed

## Next Steps

Once everything is set up:
1. Your app will now use Supabase instead of JSON files
2. All user registrations and logins are stored in Supabase
3. All detection results are saved to Supabase
4. You can view and manage data through the Supabase dashboard
5. The data is now backed up and accessible from anywhere!

## Support

If you encounter issues:
1. Check the Supabase documentation: https://supabase.com/docs
2. Check the Python client documentation: https://github.com/supabase/supabase-py
3. Verify your setup by checking the Supabase dashboard logs

