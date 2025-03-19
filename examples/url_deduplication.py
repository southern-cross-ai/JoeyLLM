import argparse
import numpy as np

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.url_dedup import (
    UrlDedupConfig,
    UrlDedupFilter,
    UrlDedupSignature,
    UrlFindDedups,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

"""
This example demonstrates URL deduplication using three pipelines:
1. Generate deduplication signatures from input data.
2. Identify duplicate URLs using the generated signatures.
3. Filter the original data by removing duplicates.
"""

# Configure URL deduplication parameters.
# - document_priority: Keeps the longest document per URL.
# - url_normalizer: Standardizes URLs (here, converting to lowercase).
url_dedup_config = UrlDedupConfig(
    document_priority=lambda doc: min(np.iinfo(np.uint16).max, len(doc.text) // 4),
    url_normalizer=lambda url: url.lower(),
)

FINDER_WORKERS = 4  # Parallelize the duplicate-finding stage.
LIMIT = -1  # Use all documents (for testing purposes).

def run_example(args):
    # Pipeline 1: Generate URL signatures.
    pipeline_1 = [
        JsonlReader(args.input_folder, limit=LIMIT, doc_progress=True),
        UrlDedupSignature(
            output_folder=f"{args.sigs_dup_folder}/sigs",
            config=url_dedup_config,
            finder_workers=FINDER_WORKERS,
        ),
    ]

    # Pipeline 2: Find duplicate URLs based on signatures.
    pipeline_2 = [
        UrlFindDedups(
            data_folder=f"{args.sigs_dup_folder}/sigs",
            output_folder=f"{args.sigs_dup_folder}/dups",
            config=url_dedup_config,
        )
    ]

    # Pipeline 3: Filter the original data by removing duplicates.
    pipeline_3 = [
        JsonlReader(data_folder=args.input_folder, limit=LIMIT, doc_progress=True),
        UrlDedupFilter(
            data_folder=f"{args.sigs_dup_folder}/dups",
            config=url_dedup_config,
            exclusion_writer=JsonlWriter(output_folder=f"{args.base_output_folder}/removed"),
        ),
        JsonlWriter(output_folder=f"{args.base_output_folder}/output"),
    ]

    # Create local executors for each pipeline stage.
    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, tasks=4)
    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, tasks=FINDER_WORKERS)
    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, tasks=4)

    # Run each pipeline sequentially.
    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())

# Parse command-line arguments and run the example.
parser = argparse.ArgumentParser(description="URL Deduplication")
parser.add_argument("input_folder", help="Input folder path")
parser.add_argument("base_output_folder", help="Base output folder path")
parser.add_argument("sigs_dup_folder", help="sigs-dup folder path")
if __name__ == "__main__":
    args = parser.parse_args()
    run_example(args)
