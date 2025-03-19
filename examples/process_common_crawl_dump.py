"""
This script performs base processing of Common Crawl data for a given dump.
The dump identifier is provided as a command-line argument (e.g., "CC-MAIN-2023-23").

Processing steps include:
1. Reading WARC files from the specified Common Crawl dump.
2. Filtering URLs to remove unwanted entries.
3. Extracting text content using Trafilatura.
4. Filtering out non-English documents.
5. Removing documents with repetitive content.
6. Applying a quality filter to remove low-quality content.
7. Writing the final processed data to JSONL files.

The pipeline is executed using SLURM for parallel processing.
"""

import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

# -----------------------------------------------------------------------------
# Ensure that a dump name is provided as a command-line argument.
# -----------------------------------------------------------------------------
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]  # e.g., "CC-MAIN-2023-23"

# Base output path on S3 where processed data and logs will be stored.
MAIN_OUTPUT_PATH = "s3://some_s3_bucket/base_processing/"

# -----------------------------------------------------------------------------
# Create a SLURM pipeline executor to process the dump.
# -----------------------------------------------------------------------------
executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP}",  # Job name includes the dump identifier.
    pipeline=[
        # Read WARC files from the given Common Crawl dump on S3.
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP}/segments/",
            glob_pattern="*/warc/*",  # Only process files matching the WARC pattern.
            default_metadata={"dump": DUMP},
        ),
        # Filter out unwanted URLs; excluded URLs are written to a separate JSONL file.
        URLFilter(
            exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/url/{DUMP}")
        ),
        # Extract textual content from the WARC files using Trafilatura.
        Trafilatura(favour_precision=True),
        # Filter documents based on language (e.g., keep only English).
        # Excluded documents are written to language-specific folders.
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                f"{MAIN_OUTPUT_PATH}/non_english/",
                output_filename="${language}/" + DUMP + "/${rank}.jsonl.gz",  # Folder structure: language/dump/file
            )
        ),
        # Remove documents with high levels of repetition.
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/repetitive/{DUMP}")
        ),
        # Apply a quality filter to remove low-quality content.
        GopherQualityFilter(
            exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/quality/{DUMP}")
        ),
        # Write the final processed output to JSONL format.
        JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DUMP}"),
    ],
    tasks=8000,                    # Total number of tasks for parallel processing.
    time="10:00:00",               # Maximum allowed time per job.
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP}",
    slurm_logs_folder=f"logs/process_dump/processing/base_processing/{DUMP}/slurm_logs",  # Local folder for SLURM logs.
    randomize_start_duration=180,  # Random delay (in seconds) to avoid overloading S3.
    mem_per_cpu_gb=2,              # Memory allocation per CPU.
    partition="hopper-cpu",        # SLURM partition to run the job.
)

# Run the pipeline to start processing.
executor.run()
