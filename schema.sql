-- Enable pgvector extension for embeddings
create extension if not exists vector;

-- Users Table
create table if not exists users (
  id uuid default gen_random_uuid() primary key,
  username text unique not null,
  email text unique not null,
  password text not null, -- Store hashed password
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Detection Results Table
create table if not exists detection_results (
  id uuid default gen_random_uuid() primary key,
  filename text not null,
  condition text not null, -- 'Healthy', 'Diseased', 'Not Pechay'
  confidence float not null,
  image_path text not null,
  recommendations jsonb, -- Stores the specific tips generated at that time
  all_probabilities jsonb,
  user_id uuid references users(id),
  timestamp timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Dataset Table (Hybrid Architecture)
create table if not exists petchay_dataset (
  id uuid default gen_random_uuid() primary key,
  filename text not null,
  condition text not null, -- 'Healthy' or 'Diseased'
  disease_name text, -- Specific disease name e.g., 'Blackrot'
  image_url text not null,
  embedding vector(512), -- Adjust dimensions based on your CNN (ResNet18 usually 512)
  user_id uuid references users(id),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Disease Information & Treatments Table (Dynamic "Cures")
create table if not exists disease_info (
  id uuid default gen_random_uuid() primary key,
  disease_name text unique not null, -- e.g., 'Blackrot', 'Soft Rot', 'Healthy'
  title text not null, -- Display title
  description text,
  tips text[], -- Array of treatment tips
  action text, -- Main action to take
  urgency text default 'low' -- 'high', 'medium', 'low'
);

-- Insert Default Data for "Cures"
insert into disease_info (disease_name, title, description, tips, action, urgency)
values 
(
  'Blackrot', 
  '⚠️ Alternaria Blackrot Detected', 
  'Black rot is a disease caused by the bacterium Xanthomonas campestris pv. campestris. It affects crucifers such as cabbage, broccoli, cauliflower, kale, turnip, and mustard.',
  ARRAY[
    'Isolate the affected plant immediately to prevent spread.',
    'Remove and destroy infected leaves (do not compost).',
    'Improve air circulation around plants.',
    'Apply copper-based fungicide if the infection is early.',
    'Avoid overhead watering to keep leaves dry.'
  ],
  'Immediate isolation and treatment required.',
  'high'
),
(
  'Healthy',
  '🌱 Your pechay leaf is healthy!',
  'The plant shows no signs of disease.',
  ARRAY[
    'Continue your current care routine - it''s working well.',
    'Maintain consistent watering (1-2 times per week).',
    'Ensure 6-8 hours of sunlight daily.',
    'Monitor for pests regularly.'
  ],
  'Keep up the excellent work! Your pechay is thriving.',
  'low'
),
(
  'Soft Rot',
  '⚠️ Soft Rot Detected',
  'Bacterial soft rot is caused by several types of bacteria that break down plant tissues.',
  ARRAY[
    'Remove infected plants immediately and destroy them.',
    'Disinfect tools used on infected plants.',
    'Avoid planting in wet, poorly drained soil.',
    'Rotate crops to prevent bacteria buildup in soil.'
  ],
  'Remove infected plants immediately.',
  'high'
)
on conflict (disease_name) do nothing;

-- Function to match embeddings (Hybrid Matcher)
create or replace function match_petchay_embeddings (
  query_embedding vector(512),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  disease_name text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    petchay_dataset.id,
    petchay_dataset.disease_name,
    1 - (petchay_dataset.embedding <=> query_embedding) as similarity
  from petchay_dataset
  where 1 - (petchay_dataset.embedding <=> query_embedding) > match_threshold
  order by petchay_dataset.embedding <=> query_embedding
  limit match_count;
end;
$$;
