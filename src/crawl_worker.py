import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_supabase_client,
    add_documents_to_supabase,
    extract_code_blocks,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    smart_chunk_markdown,
    extract_section_info
)


from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from urllib.parse import urlparse
import os
import concurrent.futures

def process_code_example(args):
    """Process a single code example to generate its summary."""
    from utils import generate_code_example_summary
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)

async def crawl_single_page_impl(url: str) -> Dict[str, Any]:
    """
    Implementation of single page crawling.
    
    Args:
        url: URL to crawl
        
    Returns:
        Dictionary with crawl results
    """
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        supabase_client = get_supabase_client()
        
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            chunks = smart_chunk_markdown(result.markdown)
            
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                metadatas.append(meta)
                
                total_word_count += meta.get("word_count", 0)
            
            url_to_full_document = {url: result.markdown}
            
            # Update source info
            source_summary = extract_source_summary(source_id, result.markdown[:5000])
            update_source_info(supabase_client, source_id, source_summary, total_word_count)
            
            # Add documents
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            # Process code examples if enabled
            code_blocks = []
            if os.getenv("USE_AGENTIC_RAG", "false") == "true":
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
                    
                    add_code_examples_to_supabase(
                        supabase_client, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas
                    )
            
            return {
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks),
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id
            }
        else:
            return {
                "success": False,
                "url": url,
                "error": result.error_message
            }

async def smart_crawl_url_impl(url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> Dict[str, Any]:
    """
    Implementation of smart crawling.
    
    Args:
        url: URL to crawl
        max_depth: Maximum recursion depth
        max_concurrent: Maximum concurrent requests
        chunk_size: Chunk size for content
        
    Returns:
        Dictionary with crawl results
    """
    # Import crawling functions from main module
    from crawl4ai_mcp import (
        is_txt, is_sitemap, parse_sitemap,
        crawl_markdown_file, crawl_batch, crawl_recursive_internal_links
    )
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        supabase_client = get_supabase_client()
        
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = parse_sitemap(url)
            if sitemap_urls:
                crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
                crawl_type = "sitemap"
        else:
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return {
                "success": False,
                "url": url,
                "error": "No content found"
            }
        
        # Process and store results (similar to crawl_single_page but for multiple pages)
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        source_content_map = {}
        source_word_counts = {}
        
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                metadatas.append(meta)
                
                source_word_counts[source_id] += meta.get("word_count", 0)
                chunk_count += 1
        
        url_to_full_document = {doc['url']: doc['markdown'] for doc in crawl_results}
        
        # Update sources
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))
        
        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)
        
        # Add documents
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=20)
        
        # Process code examples if enabled
        code_examples_count = 0
        if os.getenv("USE_AGENTIC_RAG", "false") == "true":
            # Similar code example processing as single page...
            pass
        
        return {
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": code_examples_count,
            "sources_updated": len(source_content_map)
        }

def process_crawl_job(job_type: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a crawl job from the Redis queue.
    
    Args:
        job_type: Type of job ('single_page' or 'smart_crawl')
        job_data: Job parameters
        
    Returns:
        Job result dictionary
    """
    try:
        if job_type == "single_page":
            result = asyncio.run(crawl_single_page_impl(job_data["url"]))
        elif job_type == "smart_crawl":
            result = asyncio.run(smart_crawl_url_impl(
                url=job_data["url"],
                max_depth=job_data.get("max_depth", 3),
                max_concurrent=job_data.get("max_concurrent", 10),
                chunk_size=job_data.get("chunk_size", 5000)
            ))
        else:
            result = {"success": False, "error": f"Unknown job type: {job_type}"}
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "job_type": job_type,
            "url": job_data.get("url", "unknown")
        }

if __name__ == "__main__":
    """
    Run the worker process.
    Usage: python crawl_worker.py
    """
    from rq import Worker
    from redis_queue import redis_conn, crawl_queue
    
    print("Starting crawl worker...")
    print(f"Listening to queue: {crawl_queue.name}")
    
    worker = Worker([crawl_queue], connection=redis_conn)
    worker.work()