import polars as pl
from urllib.parse import urlparse
from huggingface_hub import list_repo_files
import fsspec
from tqdm import tqdm
from pathlib import Path
import tempfile
import shutil
import logging
import multiprocessing
from multiprocessing import Pool

# ========== CONFIG ==========
REPO_ID = "HuggingFaceFW/fineweb"   # Repository ID on Hugging Face
SUBFOLDER = "data/" # USING main DATA
# for sample data use "sample/100BT" 
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"   # Base URL for remote files
OUTPUT_DIR = Path("output") # LOCATION OF FILTERED FILES 
MAX_FILES = 1000000 # Limit number of files to process
# ============================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(Path("output") / "au_filter.log"), logging.StreamHandler()]
)

def process_file(rel_path):
    remote_url = BASE_URL + rel_path
    fname = Path(rel_path).name
    subdir = Path(rel_path).parent
    out_subdir = OUTPUT_DIR / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)
    output_path = out_subdir / fname.replace(".parquet", "_au.parquet")

    if output_path.exists():
        return

    logging.info(f"📄 Processing file: {rel_path}")

    try:
        temp_path = tempfile.mktemp(suffix=".parquet")
        with fsspec.open(remote_url, mode='rb') as remote_file, open(temp_path, "wb") as local_file:
            shutil.copyfileobj(remote_file, local_file)

        lf = pl.scan_parquet(temp_path)

        lf = lf.filter(pl.col("url").str.contains("\\."))

        lf = lf.with_columns([
            pl.col("url").map_elements(lambda x: (urlparse(x).netloc.lower().replace("www.", "").split(":")[0] if urlparse(x).netloc else "") if x else "", return_dtype=pl.Utf8).alias("domain"),
            pl.col("url").map_elements(lambda x: urlparse(x).path.lower() if x else "", return_dtype=pl.Utf8).alias("path")
        ])

        df = lf.collect()

        df_filtered = df.filter(
            (pl.col("domain").str.ends_with(".au")) |
            (pl.col("path").str.contains(r"(^|/)au(/|$)", literal=False)) |
            (pl.col("path").str.contains(r"(^|/)en-au(/|$)", literal=False))
        )

        if df_filtered.shape[0] > 0:
            df_filtered.write_parquet(output_path)
            logging.info(f"✅ Saved {df_filtered.shape[0]} AU rows to {output_path.name}")
        else:
            logging.info(f"⚠️ No AU rows in {output_path.name}")

    except Exception as e:
        logging.error(f"❌ Error processing {remote_url}: {e}")
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


# ========== MAIN SCRIPT ==========

logging.info("📡 Listing files from Hugging Face...")
all_files = list_repo_files(REPO_ID, repo_type="dataset")
parquet_files = [f for f in all_files if f.startswith(SUBFOLDER) and f.endswith(".parquet")]

if MAX_FILES:
    parquet_files = parquet_files[:MAX_FILES]

logging.info(f"📁 Found {len(parquet_files)} files to process")
logging.info(f"⚠️ Using {multiprocessing.cpu_count()} available CPU cores for processing.")
with Pool(processes=multiprocessing.cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(process_file, parquet_files), total=len(parquet_files), desc="Processing files"))
