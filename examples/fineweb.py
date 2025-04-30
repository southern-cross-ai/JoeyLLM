"""
This file contains the code used to process and create the
FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb).

The processing is divided into two main parts:
1. Base Processing: Reads raw WARC files from Common Crawl, applies a series of
   filtering and extraction steps, and outputs cleaned JSONL files.
2. Minhash Deduplication: Uses a multi-stage pipeline to compute minhash signatures,
   bucket similar documents, cluster duplicates, and finally filter out the duplicate content.
"""

# =============================================================================
# Import necessary modules and classes for SLURM pipeline execution and processing steps
# =============================================================================
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig


# =============================================================================
# Define dataset and output paths
# =============================================================================
DUMP_TO_PROCESS = "CC-MAIN-2023-50"  # Example dump identifier

MAIN_OUTPUT_PATH = "s3://some_s3_bucket"
FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"


# =============================================================================
# BASE PROCESSING PIPELINE
# =============================================================================
"""
The first pipeline processes the raw WARC files from Common Crawl.
It performs the following steps:
1. Reads WARC files from an S3 bucket.
2. Filters URLs to remove unwanted ones.
3. Extracts the text from HTML content using Trafilatura.
4. Filters documents by language (only English is kept).
5. Removes documents with repetitive content (Gopher repetition filter).
6. Applies quality filters (Gopher and C4) to ensure high-quality data.
7. Applies a specific quality filter for FineWeb requirements.
8. Writes the final processed documents to JSONL files.
"""

main_processing_executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP_TO_PROCESS}",
    pipeline=[
        # Read WARC files from the Common Crawl S3 bucket.
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",  # Only include WARC files.
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        # Filter URLs and log exclusions.
        URLFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")
        ),
        # Extract text using Trafilatura; favouring precision for better extraction.
        Trafilatura(favour_precision=True),
        # Filter out non-English documents, writing exclusions to a structured folder.
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                f"{FILTERING_OUTPUT_PATH}/2_non_english/",
                output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
            )
        ),
        # Remove repetitive content using the Gopher repetition filter.
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")
        ),
        # Filter based on Gopher quality metrics.
        GopherQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")
        ),
        # Apply a C4 quality filter (with terminal punctuation settings).
        C4QualityFilter(
            filter_no_terminal_punct=False,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
        ),
        # Filter documents to meet FineWeb-specific quality requirements.
        FineWebQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")
        ),
        # Write the final, processed output.
        JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
    ],
    tasks=8000,                   # Total tasks for parallel execution.
    time="10:00:00",              # Maximum job runtime.
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
    slurm_logs_folder=f"logs/base_processing/{DUMP_TO_PROCESS}/slurm_logs",  # Local SLURM log directory.
    randomize_start_duration=180, # Random delay to avoid overwhelming S3 with requests.
    mem_per_cpu_gb=2,
    partition="hopper-cpu",
)

# Run the base processing pipeline.
main_processing_executor.run()


# =============================================================================
# MINHASH DEDUPLICATION PIPELINE
# =============================================================================
"""
After base processing, we apply minhash deduplication to remove duplicate documents.
This deduplication process consists of four stages:

Stage 1: Compute Minhash Signatures
    - Each task computes minhash signatures for a set of documents, using SHA-1 for hashing.
Stage 2: Bucket the Minhash Signatures
    - Organize signatures into buckets based on similarity to facilitate duplicate lookup.
Stage 3: Cluster Duplicates
    - Cluster documents that appear similar based on bucketed minhash signatures.
Stage 4: Filter Out Duplicates and Finalize Dataset
    - Remove duplicate documents, perform token counting for reporting, format PII, and write the final deduplicated output.
"""

# -------------------
# Define minhash configuration parameters
# -------------------
minhash_config = MinhashConfig(
    hash_config=HashConfig(
        hash_fc="sha1",  # Using SHA-1 for higher precision (fewer collisions).
        precision=64,
    ),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

# S3 paths for storing minhash data and logs.
S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"
S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
LOCAL_LOGS_FOLDER = "logs/minhash"

TOTAL_TASKS = 1000  # Total tasks for parallel processing in minhash stages.

# The input reader for minhash stages: reads the processed data from the base processing output.
INPUT_READER = JsonlReader(
    f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"
)


# -------------------
# Stage 1: Compute Minhash Signatures
# -------------------
# Compute a minhash signature for each document (or batch of documents).
stage1 = SlurmPipelineExecutor(
    job_name=f"mh1_{DUMP_TO_PROCESS}",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
            config=minhash_config
        ),
    ],
    tasks=TOTAL_TASKS,
    time="5:00:00",
    partition="hopper-cpu",
    logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
    randomize_start_duration=180,
    depends=main_processing_executor,  # This stage starts after the base processing completes.
)


# -------------------
# Stage 2: Bucket the Minhash Signatures
# -------------------
# Organize the computed minhash signatures into buckets for efficient duplicate lookup.
stage2 = SlurmPipelineExecutor(
    job_name=f"mh2_{DUMP_TO_PROCESS}",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
            config=MinhashConfig(hash_config=minhash_config.hash_config),
        ),
    ],
    tasks=minhash_config.num_buckets * 50,  # Run 50 workers per bucket.
    randomize_start_duration=180,
    logging_dir=f"{S3_LOGS_FOLDER}/buckets",
    partition="hopper-cpu",
    time="02:00:00",
    mem_per_cpu_gb=4,
    cpus_per_task=3,  # Adjust task parameters if memory is limited.
    depends=stage1,
)


# -------------------
# Stage 3: Cluster Duplicates Using Minhash Buckets
# -------------------
# Cluster similar documents into duplicate groups based on the minhash buckets.
stage3 = SlurmPipelineExecutor(
    job_name=f"mh3_{DUMP_TO_PROCESS}",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,  # Runs as a single task.
    logging_dir=f"{S3_LOGS_FOLDER}/clustering",
    partition="hopper-cpu",
    time="30:00:00",  # Clustering may take longer.
    mem_per_cpu_gb=25,
    cpus_per_task=8,  # High resource allocation needed for clustering.
    depends=stage2,
)


# -------------------
# Stage 4: Filter Duplicates and Finalize Dataset
# -------------------
# Filter out duplicate documents using the clusters from Stage 3.
# Additionally, count tokens (for before/after metrics), format PII, and output the final deduplicated dataset.
stage4 = SlurmPipelineExecutor(
    job_name=f"mh4_{DUMP_TO_PROCESS}",
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # Counts tokens to assess the impact of deduplication.
        MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids"),
        PIIFormatter(),  # Remove or format Personally Identifiable Information.
        JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/deduped_output"),
    ],
    tasks=TOTAL_TASKS,
    logging_dir=f"{S3_LOGS_FOLDER}/filtering",
    partition="hopper-cpu",
    time="5:00:00",
    mem_per_cpu_gb=4,
    depends=stage3,
)

# Launch the final deduplication stage.
stage4.run()
