
create extension if not exists vector;

-- Drop tables if they exist (to allow rerunning the script)
drop table if exists crawled_pages cascade;
drop table if exists code_examples cascade;
drop table if exists sources cascade;
drop table if exists crawl_jobs cascade;
drop table if exists url_processing_log cascade;

-- ============================================
-- JOB TRACKING TABLE
-- ============================================
create table crawl_jobs (
    job_id varchar(255) primary key,
    job_type varchar(50) not null check (job_type in ('single_page', 'smart_crawl')),
    url text not null,
    status varchar(20) not null default 'pending' 
        check (status in ('pending', 'queued', 'started', 'finished', 'failed', 'canceled')),
    
    -- Timestamps
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    queued_at timestamp with time zone,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    
    -- Job details
    parameters jsonb not null default '{}'::jsonb,
    error_message text,
    result_summary jsonb,
    
    -- Progress tracking
    pages_processed integer default 0,
    chunks_created integer default 0,
    code_examples_created integer default 0,
    
    -- Retry logic
    retry_count integer default 0,
    max_retries integer default 3,
    
    -- Worker info
    worker_id varchar(100),
    processing_time_ms bigint
);

-- Indexes for job tracking
create index idx_crawl_jobs_status on crawl_jobs(status);
create index idx_crawl_jobs_created on crawl_jobs(created_at desc);
create index idx_crawl_jobs_url on crawl_jobs(url);
create index idx_crawl_jobs_type_status on crawl_jobs(job_type, status);

-- ============================================
-- URL DEDUPLICATION & RATE LIMITING
-- ============================================
create table url_processing_log (
    id bigserial primary key,
    url text not null,
    url_hash varchar(64) not null unique,
    last_crawled_at timestamp with time zone default timezone('utc'::text, now()) not null,
    crawl_count integer default 1,
    last_job_id varchar(255),
    source_id text,
    
    foreign key (last_job_id) references crawl_jobs(job_id) on delete set null
);

create index idx_url_hash on url_processing_log(url_hash);
create index idx_last_crawled on url_processing_log(last_crawled_at desc);

-- ============================================
-- SOURCES TABLE
-- ============================================
create table sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    total_pages integer default 0,
    total_code_examples integer default 0,
    last_crawled_at timestamp with time zone,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create index idx_sources_last_crawled on sources(last_crawled_at);

-- ============================================
-- CRAWLED PAGES TABLE
-- ============================================
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),
    job_id varchar(255),
    content_hash varchar(64),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id) on delete cascade,
    foreign key (job_id) references crawl_jobs(job_id) on delete set null
);

create index on crawled_pages using ivfflat (embedding vector_cosine_ops);
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);
create index idx_crawled_pages_source_id on crawled_pages (source_id);
create index idx_crawled_pages_job_id on crawled_pages (job_id);
create index idx_crawled_pages_hash on crawled_pages (content_hash);

-- ============================================
-- CODE EXAMPLES TABLE
-- ============================================
create table code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    summary text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),
    job_id varchar(255),
    language varchar(50),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id) on delete cascade,
    foreign key (job_id) references crawl_jobs(job_id) on delete set null
);

create index on code_examples using ivfflat (embedding vector_cosine_ops);
create index idx_code_examples_metadata on code_examples using gin (metadata);
create index idx_code_examples_source_id on code_examples (source_id);
create index idx_code_examples_job_id on code_examples (job_id);
create index idx_code_examples_language on code_examples (language);

-- ============================================
-- SEARCH FUNCTIONS
-- ============================================
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    source_id,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

create or replace function match_code_examples (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    summary,
    metadata,
    source_id,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  from code_examples
  where metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  order by code_examples.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- ============================================
-- UTILITY FUNCTIONS
-- ============================================
create or replace function is_url_recently_crawled(
    url_to_check text,
    hours_threshold integer default 24
) returns boolean
language plpgsql
as $$
declare
    url_hash_val varchar(64);
    last_crawl timestamp with time zone;
begin
    url_hash_val := encode(digest(url_to_check, 'sha256'), 'hex');
    
    select last_crawled_at into last_crawl
    from url_processing_log
    where url_hash = url_hash_val;
    
    if last_crawl is null then
        return false;
    end if;
    
    return (now() - last_crawl) < interval '1 hour' * hours_threshold;
end;
$$;

create or replace function get_job_stats(
    time_window_hours integer default 24
) returns table (
    total_jobs bigint,
    pending_jobs bigint,
    running_jobs bigint,
    completed_jobs bigint,
    failed_jobs bigint,
    avg_processing_time_ms numeric,
    total_chunks_created bigint,
    total_pages_processed bigint
)
language plpgsql
as $$
begin
    return query
    select
        count(*)::bigint as total_jobs,
        count(*) filter (where status = 'pending')::bigint as pending_jobs,
        count(*) filter (where status in ('queued', 'started'))::bigint as running_jobs,
        count(*) filter (where status = 'finished')::bigint as completed_jobs,
        count(*) filter (where status = 'failed')::bigint as failed_jobs,
        avg(processing_time_ms)::numeric as avg_processing_time_ms,
        sum(chunks_created)::bigint as total_chunks_created,
        sum(pages_processed)::bigint as total_pages_processed
    from crawl_jobs
    where created_at > now() - interval '1 hour' * time_window_hours;
end;
$$;

create or replace function cleanup_old_jobs(
    days_to_keep integer default 30
) returns integer
language plpgsql
as $$
declare
    deleted_count integer;
begin
    delete from crawl_jobs
    where completed_at < now() - interval '1 day' * days_to_keep
      and status in ('finished', 'failed', 'canceled');
    
    get diagnostics deleted_count = row_count;
    return deleted_count;
end;
$$;

-- ============================================
-- TRIGGERS
-- ============================================
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

create trigger update_sources_updated_at
    before update on sources
    for each row
    execute function update_updated_at_column();

create trigger update_crawled_pages_updated_at
    before update on crawled_pages
    for each row
    execute function update_updated_at_column();

create trigger update_code_examples_updated_at
    before update on code_examples
    for each row
    execute function update_updated_at_column();

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
alter table crawled_pages enable row level security;
alter table code_examples enable row level security;
alter table sources enable row level security;
alter table crawl_jobs enable row level security;
alter table url_processing_log enable row level security;

create policy "Allow public read access to crawled_pages"
    on crawled_pages for select to public using (true);

create policy "Allow public read access to code_examples"
    on code_examples for select to public using (true);

create policy "Allow public read access to sources"
    on sources for select to public using (true);

create policy "Allow public read access to crawl_jobs"
    on crawl_jobs for select to public using (true);

create policy "Allow public read access to url_processing_log"
    on url_processing_log for select to public using (true);