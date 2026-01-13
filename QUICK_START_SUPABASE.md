# Quick Start: Supabase Integration

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Supabase Credentials
1. Go to [supabase.com](https://supabase.com) and create a project
2. In Project Settings > API, copy:
   - **Project URL**
   - **anon public key**

### 3. Create `.env` File
```bash
cp config_template.env .env
```

Edit `.env` and add your credentials:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
FLASK_SECRET_KEY=your-random-secret-key
```

### 4. Create Database Tables
1. Open Supabase Dashboard > SQL Editor
2. Copy and run all SQL from `supabase_schema.sql`
3. Click "Run" to create tables

### 5. Migrate Existing Data (Optional)
```bash
python migrate_to_supabase.py
```

### 6. Start Your App
```bash
python app.py
```

## ✅ What Changed?

- ✅ Users now stored in Supabase `users` table
- ✅ Detection results stored in Supabase `detection_results` table
- ✅ All queries use Supabase instead of JSON files
- ✅ Dashboard stats calculated from Supabase
- ✅ Results page displays from Supabase

## 📁 Files Added

- `db.py` - Database helper functions
- `supabase_schema.sql` - Database table definitions
- `migrate_to_supabase.py` - Migration script
- `config_template.env` - Environment variable template
- `SUPABASE_SETUP.md` - Detailed setup guide

## 🔧 Files Modified

- `app.py` - Updated to use Supabase instead of JSON
- `requirements.txt` - Added supabase and python-dotenv

## 💡 Next Steps

1. Test user registration and login
2. Upload an image and verify it's saved to Supabase
3. Check Supabase dashboard > Table Editor to see your data
4. Delete old JSON files after verifying everything works

## 🆘 Troubleshooting

**"Missing Supabase credentials"**
- Make sure `.env` file exists (not just `config_template.env`)
- Restart Flask after creating `.env`

**"Table does not exist"**
- Run `supabase_schema.sql` in Supabase SQL Editor

**Migration fails**
- Check your internet connection
- Verify credentials are correct
- Make sure tables are created first

For more details, see `SUPABASE_SETUP.md`

