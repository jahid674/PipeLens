import ast
import pandas as pd
import os
import time
import tqdm
import gc
import threading
from queue import Queue
import concurrent.futures

def process_chunk(lines):
    """Process a chunk of lines and return valid products."""
    products = []
    for line in lines:
        if not line.strip():
            continue
            
        try:
            item = ast.literal_eval(line.strip())
            
            # Skip items without a brand
            if 'brand' not in item or not item['brand']:
                continue
                
            # Extract only the fields we need (minimal data)
            product = {
                'asin': item.get('asin'),
                'title': item.get('title'),
                'price': item.get('price', None),
                'brand': item.get('brand'),
                'categories': item.get('categories', None)
            }
            products.append(product)
        except (SyntaxError, ValueError):
            continue
    
    return products

def producer(filename, queue, chunk_size=10000):
    """Read file in chunks and add to queue."""
    with open(filename, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if len(lines) >= chunk_size:
                queue.put(lines)
                lines = []
        # Put remaining lines
        if lines:
            queue.put(lines)
    # Signal end of file
    queue.put(None)

def process_large_file(file_path, output_path="AmazonData/products_processed.csv", 
                      chunk_size=10000, num_workers=4, 
                      brands_per_file=1000):
    """
    Process a very large file of product data efficiently.
    
    Args:
        file_path: Path to input file
        output_path: Path to save results
        chunk_size: Number of lines to process at once
        num_workers: Number of worker threads
        brands_per_file: Number of brands to save per output file
    """
    start_time = time.time()
    
    # Count lines in file first
    print("Counting lines in file...")
    num_lines = 0
    with open(file_path, 'r') as f:
        for _ in f:
            num_lines += 1
    
    print(f"File contains {num_lines:,} lines. Starting processing...")
    
    # Set up producer-consumer pattern
    queue = Queue(maxsize=num_workers * 2)  # Buffer some chunks
    
    # Start producer thread
    producer_thread = threading.Thread(
        target=producer, 
        args=(file_path, queue, chunk_size)
    )
    producer_thread.start()
    
    # Process chunks with thread pool
    all_products = []
    brands_seen = set()
    chunk_counter = 0
    file_counter = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        done = False
        
        # Progress bar for overall process
        with tqdm.tqdm(total=num_lines, desc="Processing", unit="lines") as pbar:
            while not done or futures:
                # Submit new tasks
                while not done and len(futures) < num_workers * 2:
                    chunk = queue.get()
                    if chunk is None:
                        done = True
                        break
                    
                    future = executor.submit(process_chunk, chunk)
                    futures.append((future, len(chunk)))
                
                # Check for completed futures
                for i in range(len(futures) - 1, -1, -1):
                    future, size = futures[i]
                    if future.done():
                        chunk_products = future.result()
                        all_products.extend(chunk_products)
                        pbar.update(size)
                        
                        # Remove processed future
                        futures.pop(i)
                        
                        # Increment counter
                        chunk_counter += 1
                        
                        # Periodically save results and free memory
                        if chunk_counter % 100 == 0:
                            print(f"\nProcessed {chunk_counter} chunks, found {len(all_products):,} products")
                            print(f"Current memory usage: {get_memory_usage():.2f} MB")
                            
                            # Convert to DataFrame and deduplicate
                            if all_products:
                                print("Converting to DataFrame and deduplicating...")
                                df = pd.DataFrame(all_products)
                                
                                # Add new brands to brands_seen set
                                new_brands = set(df['brand'].unique()) - brands_seen
                                brands_seen.update(new_brands)
                                
                                # Filter to keep only products with new brands
                                df = df[df['brand'].isin(new_brands)]
                                
                                # Save intermediate results if we have enough new brands
                                if len(new_brands) >= brands_per_file:
                                    save_path = f"{os.path.splitext(output_path)[0]}_{file_counter}.csv"
                                    df.to_csv(save_path, index=False)
                                    print(f"Saved {len(df):,} products with {len(new_brands):,} brands to {save_path}")
                                    file_counter += 1
                                
                                # Clear memory
                                all_products = []
                                gc.collect()
                
                # Brief pause to avoid CPU spinning
                if futures:
                    time.sleep(0.01)
    
    # Convert remaining products to DataFrame
    if all_products:
        print("\nProcessing final batch...")
        df = pd.DataFrame(all_products)
        
        # Add new brands to brands_seen set
        new_brands = set(df['brand'].unique()) - brands_seen
        brands_seen.update(new_brands)
        
        # Filter to keep only products with new brands
        df = df[df['brand'].isin(new_brands)]
        
        # Save final results
        save_path = f"{os.path.splitext(output_path)[0]}_{file_counter}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df):,} products with {len(new_brands):,} brands to {save_path}")
    
    # Print summary
    print(f"\nProcessing complete in {time.time() - start_time:.2f} seconds")
    print(f"Total unique brands found: {len(brands_seen):,}")
    
    # Combine all files if needed
    # (This step is optional and can be memory-intensive for very large datasets)
    combine_files = input("Combine all CSV files into one? (y/n): ").lower().strip() == 'y'
    if combine_files:
        combine_csv_files(os.path.splitext(output_path)[0], file_counter)

def get_memory_usage():
    """Return memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def combine_csv_files(base_path, count):
    """Combine multiple CSV files into one."""
    print("Combining CSV files...")
    dfs = []
    
    for i in range(count + 1):
        file_path = f"{base_path}_{i}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Added {file_path} ({len(df):,} rows)")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(f"{base_path}_combined.csv", index=False)
        print(f"Combined file saved with {len(combined):,} rows")
    else:
        print("No files to combine")

if __name__ == "__main__":
    file_path = 'AmazonData/metadata.json'
    
    # Process the file with optimal settings for a 9M+ item file
    chunk_size = 5000  # Smaller chunks for better memory management
    num_workers = max(1, os.cpu_count() - 1)  # Leave one CPU core free
    
    print(f"Using {num_workers} worker threads and chunk size of {chunk_size}")
    
    # Start processing
    process_large_file(file_path, chunk_size=chunk_size, num_workers=num_workers)