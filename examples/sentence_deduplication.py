from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages

"""
This example demonstrates sentence deduplication in three stages, following the approach:
"Discard all but one of any three-sentence span occurring more than once."
Stages:
1. Extraction + Filtering + Signature Generation.
2. Finding duplicate sentence spans.
3. Filtering the original data using deduplication results.
"""

# Configure sentence deduplication parameters.
sent_dedup_config = SentDedupConfig(
    n_sentences=3,
    split_sentences=True,  # Change to False to split on newlines instead.
    only_dedup_in_index=True,
    min_doc_words=50,
)

# Use multiple workers to speed up the duplicate finder stage.
FINDER_WORKERS = 10

def run_example():
    # Pipeline 1: Extract content, filter quality, write intermediate data, and generate dedup signatures.
    pipeline_1 = [
        WarcReader(data_folder="warc/", limit=1000),       # Read raw data from WARC files.
        Trafilatura(),                                     # Extract text from the raw documents.
        GopherQualityFilter(min_stop_words=0),             # Basic quality filter.
        LanguageFilter(language_threshold=0.5, languages=[Languages.english]),  # Keep mostly English documents.
        JsonlWriter("intermediate/"),                      # Write the processed output.
        SentenceDedupSignature(output_folder="c4/sigs", config=sent_dedup_config, finder_workers=FINDER_WORKERS),
    ]

    # Pipeline 2: Use signatures to identify duplicate sentence spans.
    pipeline_2 = [
        SentenceFindDedups(data_folder="c4/sigs", output_folder="c4/dups", config=sent_dedup_config)
    ]

    # Pipeline 3: Filter out duplicates from the original data and write the deduplicated output.
    pipeline_3 = [
        JsonlReader(data_folder="intermediate/"),
        SentenceDedupFilter(data_folder="c4/dups", config=sent_dedup_config),
        JsonlWriter("c4/final_output"),  # Save the final filtered data.
    ]

    # Create local executors for each pipeline stage.
    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=4, tasks=4)
    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1, tasks=FINDER_WORKERS)
    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, workers=4, tasks=4)

    # Run each pipeline sequentially.
    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())

if __name__ == "__main__":
    run_example()
