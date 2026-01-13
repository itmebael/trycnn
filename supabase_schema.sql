-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Create helper function for updating timestamps
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language 'plpgsql';

-- Create Users Table
create table public.users ( 
   id uuid not null default uuid_generate_v4 (), 
   username character varying(100) not null, 
   email character varying(255) not null, 
   password character varying(255) not null, 
   created_at timestamp with time zone null default now(), 
   updated_at timestamp with time zone null default now(), 
   constraint users_pkey primary key (id), 
   constraint users_email_key unique (email), 
   constraint users_username_key unique (username) 
 ) TABLESPACE pg_default; 
 
-- Create Indexes
create index IF not exists idx_users_username on public.users using btree (username) TABLESPACE pg_default; 
create index IF not exists idx_users_email on public.users using btree (email) TABLESPACE pg_default; 

-- Create Trigger
create trigger update_users_updated_at BEFORE 
update on users for EACH row 
execute FUNCTION update_updated_at_column ();
