import polars as pl
from urllib.parse import urlparse
from huggingface_hub import list_repo_files
import whois
import fsspec
from tqdm import tqdm
from pathlib import Path
import json
import tempfile
import shutil
import logging

# ========== CONFIG ==========
REPO_ID = "HuggingFaceFW/fineweb"   # Repository ID on Hugging Face
SUBFOLDER = "sample/100BT"  # Subfolder in the dataset
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"   # Base URL for remote files
OUTPUT_DIR = Path("output/100BT_au")    # Output directory for filtered files
ENABLE_WHOIS_LOOKUP = True  # Enable or disable WHOIS lookups
MAX_WHOIS_PER_FILE = 100   # Limit number of WHOIS lookups per file
MAX_FILES = 2   # Limit number of files to process
# ============================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WHOIS_CACHE_FILE = OUTPUT_DIR / "whois_cache.json"
AU_DOMAINS_FILE = OUTPUT_DIR / "au_domains.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(OUTPUT_DIR / "au_filter.log"), logging.StreamHandler()]
)

whois_cache = {}
au_domains = set()

if ENABLE_WHOIS_LOOKUP and WHOIS_CACHE_FILE.exists():
    with open(WHOIS_CACHE_FILE, "r") as f:
        whois_cache = json.load(f)
    logging.info(f"📦 Loaded WHOIS cache with {len(whois_cache)} entries")

if AU_DOMAINS_FILE.exists():
    with open(AU_DOMAINS_FILE, "r") as f:
        au_domains.update(json.load(f))

# ========== HELPERS ==========

def download_to_temp(remote_url: str) -> Path:
    tmp_path = tempfile.mktemp(suffix=".parquet")
    with fsspec.open(remote_url, mode='rb') as remote_file, open(tmp_path, "wb") as local_file:
        shutil.copyfileobj(remote_file, local_file)
    return Path(tmp_path)

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "").split(":")[0]
    except:
        return ""

def extract_path(url: str) -> str:
    try:
        return urlparse(url).path.lower()
    except:
        return ""

def is_au_domain(domain: str) -> bool:
    return domain.endswith(".au")

def mentions_au_path(path: str) -> bool:
    return (
        "/au/" in path or
        path.endswith("/au") or
        "/en-au/" in path or
        path.endswith("/en-au")
    )

def is_domain_australian_by_whois(domain: str) -> bool:
    if domain in whois_cache:
        return whois_cache[domain]
    try:
        info = whois.whois(domain)
        country = (info.get("country") or "").strip().lower()
        is_au = country in ("au", "australia")
    except:
        is_au = False
    whois_cache[domain] = is_au
    if is_au:
        au_domains.add(domain)
    return is_au

# ========== MAIN LOGIC ==========

def process_parquet_remote(remote_url: str, output_path: Path, use_whois=True, max_whois=100):
    try:
        temp_path = download_to_temp(remote_url)
        lf = pl.scan_parquet(temp_path)

        lf = lf.filter(pl.col("url").str.contains("\\."))

        lf = lf.with_columns([
            pl.col("url").map_elements(extract_domain, return_dtype=pl.Utf8).alias("domain"),
            pl.col("url").map_elements(extract_path, return_dtype=pl.Utf8).alias("path")
        ])

        df = lf.collect()

        df_filtered = df.filter(
            (pl.col("domain").str.ends_with(".au")) |
            (pl.col("path").str.contains(r"(^|/)au(/|$)", literal=False)) |
            (pl.col("path").str.contains(r"(^|/)en-au(/|$)", literal=False))
        )

        filtered_domains = set(df_filtered["domain"].unique())
        au_domains.update(filtered_domains)

        if use_whois and df_filtered.shape[0] < df.shape[0]:
            whois_checked = 0

            for row in df.iter_rows(named=True):
                domain = row.get("domain", "")
                if (
                    not is_au_domain(domain) and
                    not mentions_au_path(row.get("path", "")) and
                    domain not in whois_cache and
                    domain not in au_domains and
                    whois_checked < max_whois
                ):
                    if is_domain_australian_by_whois(domain):
                        df_filtered.vstack(pl.DataFrame([row]), in_place=True)
                    whois_checked += 1

        if df_filtered.shape[0] > 0:
            df_filtered.write_parquet(output_path)
            logging.info(f"✅ Saved {df_filtered.shape[0]} AU rows to {output_path.name}")
        else:
            logging.info(f"⚠️ No AU rows in {output_path.name}")

        with open(WHOIS_CACHE_FILE, "w") as f:
            json.dump(whois_cache, f)

        with open(AU_DOMAINS_FILE, "w") as f:
            json.dump(sorted(list(au_domains)), f)

    except Exception as e:
        logging.error(f"❌ Error processing {remote_url}: {e}")
    finally:
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)

# ========== MAIN SCRIPT ==========

logging.info("📡 Listing files from Hugging Face...")
all_files = list_repo_files(REPO_ID, repo_type="dataset")
parquet_files = [f for f in all_files if f.startswith(SUBFOLDER) and f.endswith(".parquet")]

if MAX_FILES:
    parquet_files = parquet_files[:MAX_FILES]

logging.info(f"📁 Found {len(parquet_files)} files to process")

for rel_path in tqdm(parquet_files, desc="Processing files"):
    remote_url = BASE_URL + rel_path
    fname = Path(rel_path).name
    output_path = OUTPUT_DIR / fname.replace(".parquet", "_au.parquet")

    if output_path.exists():
        continue

    logging.info(f"📄 Processing file: {rel_path}")
    process_parquet_remote(remote_url, output_path, use_whois=ENABLE_WHOIS_LOOKUP, max_whois=MAX_WHOIS_PER_FILE)
