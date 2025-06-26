import pandas as pd
from urllib.parse import urlparse
from huggingface_hub import list_repo_files
import whois
import fsspec
from tqdm import tqdm
from pathlib import Path
import json

# ========== CONFIG ==========
REPO_ID = "HuggingFaceFW/fineweb"
SUBFOLDER = "sample/100BT"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
OUTPUT_DIR = Path("output/100BT_au")
ENABLE_WHOIS_LOOKUP = True
MAX_WHOIS_PER_FILE = 100
MAX_FILES = 1  # Limit number of files to process; set to None for no limit
# ============================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

whois_cache = {}
if ENABLE_WHOIS_LOOKUP:
    # Global WHOIS result cache
    WHOIS_CACHE_FILE = Path("cache/whois_cache.json")
    # Load existing cache if available
    if WHOIS_CACHE_FILE.exists():
        with open(WHOIS_CACHE_FILE, "r") as f:
            whois_cache = json.load(f)
        print(f"📦 Loaded WHOIS cache with {len(whois_cache)} entries")

# ========== FILTER HELPERS ==========

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "").split(":")[0]
    except:
        return ""

def is_au_domain(domain: str) -> bool:
    return domain.endswith(".au")

def is_domain_australian_by_whois(domain: str) -> bool:
    if domain in whois_cache:
        return whois_cache[domain]
    
    try:
        info = whois.whois(domain)
        country = (info.get("country") or "").lower()
        is_au = country in ("au", "australia")
    except:
        is_au = False
    
    whois_cache[domain] = is_au
    return is_au

def filter_au_rows(df: pd.DataFrame, use_whois=True, max_whois=100):
    results = []
    whois_checked = 0

    for _, row in df.iterrows():
        url = row.get("url", "")
        if not isinstance(url, str) or not url.strip():
            continue

        domain = extract_domain(url)
        if not domain:
            continue

        if is_au_domain(domain):
            results.append(row)
        elif use_whois and whois_checked < max_whois:
            whois_checked += 1
            if is_domain_australian_by_whois(domain):
                results.append(row)

    return pd.DataFrame(results)

# ========== PROCESSING ==========

def process_parquet_remote(remote_url: str, output_path: Path, use_whois=True, max_whois=100):
    try:
        with fsspec.open(remote_url) as f:
            df = pd.read_parquet(f)

        df_au = filter_au_rows(df, use_whois=use_whois, max_whois=max_whois)

        if not df_au.empty:
            df_au.to_parquet(output_path, index=False)
            print(f"✅ Saved {len(df_au)} AU rows to {output_path.name}")
        else:
            print(f"⚠️ No AU rows in {output_path.name}")

        # Save updated WHOIS cache
        with open(WHOIS_CACHE_FILE, "w") as f:
            json.dump(whois_cache, f)

    except Exception as e:
        print(f"❌ Error processing {remote_url}: {e}")

# ========== MAIN SCRIPT ==========

print("📡 Listing files from Hugging Face...")
all_files = list_repo_files(REPO_ID, repo_type="dataset")
parquet_files = [f for f in all_files if f.startswith(SUBFOLDER) and f.endswith(".parquet")]

if MAX_FILES:
    parquet_files = parquet_files[:MAX_FILES]

print(f"📁 Found {len(parquet_files)} files to process")

for rel_path in tqdm(parquet_files, desc="Processing files"):
    remote_url = BASE_URL + rel_path
    fname = Path(rel_path).name
    output_path = OUTPUT_DIR / fname.replace(".parquet", "_au.parquet")

    if output_path.exists():
        continue

    print(f"📄 Processing file: {rel_path}")

    process_parquet_remote(remote_url, output_path, use_whois=ENABLE_WHOIS_LOOKUP, max_whois=MAX_WHOIS_PER_FILE)
