import os

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import ESDatasetToSequence, ESMergeSequences, ESRangeRemover
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages


"""
This script demonstrates how to perform exact-substring deduplication using a combination of DataTrove pipelines
and Google's deduplicate-text-datasets. Here is the general flow:

1) ESDatasetToSequence (Stage 1): 
   - Reads each input file and maps its contents into a sequence, inserting unique separators 
     before each document.
   - Records the byte offsets indicating where each document starts.

2) ESMergeSequences (Stage 2):
   - Merges all individual sequences from stage 1 into a single large sequence.
   - Also saves the byte offsets for each file.

3) (External step) Use the deduplicate-text-datasets scripts to:
   - Create a suffix array of the merged sequence.
   - Identify duplicates and produce a .bytearange file containing duplicate byte ranges.

4) ESRangeRemover (Stage 3):
   - Reads the original documents (via a pipeline).
   - Leverages the .bytearange file to remove or skip duplicate ranges.
   - Writes out the deduplicated documents.

Workflow:
- run_step_1_and_2()
    - Executes the pipelines for stages 1 and 2, ending with a single merged file.
- After that completes, follow deduplicate-text-datasets instructions to detect duplicates.
- run_step_3()
    - Uses ESRangeRemover to drop duplicates based on the range file and writes the final deduplicated dataset.

"""


def run_step_1_and_2():
    """
    Stage 1 and 2:
    1) Read WARC files (limit=1000 in this example), parse them with Trafilatura, and filter by quality and language.
    2) Write the filtered data to a JSONL file.
    3) Convert these JSONL documents into sequences (ESDatasetToSequence).
    4) Merge all sequences into a single large sequence file (ESMergeSequences).
    """
    pipeline_1 = [
        WarcReader(data_folder="warc/", limit=1000),
        Trafilatura(),  # Extract readable text from HTML
        GopherQualityFilter(min_stop_words=0),  # Filter based on a quality metric
        LanguageFilter(language_threshold=0.5, languages=[Languages.english]),  # Keep English text
        JsonlWriter("intermediate/"),  # Write intermediate filtered documents
        ESDatasetToSequence(output_folder="es/"),  # Convert documents to sequences
    ]

    # Merges all sequences produced above into one large sequence
    pipeline_2 = [
        ESMergeSequences(
            data_folder="es",
            tasks_stage_1=4,
        )
    ]

    # Create local executors to run each pipeline
    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=4, tasks=4)
    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1, tasks=1)

    print(executor_1.run())
    print(executor_2.run())


def run_step_3():
    """
    Stage 3:
    1) Read the intermediate JSONL data.
    2) Use ESRangeRemover to remove duplicates identified by deduplicate-text-datasets in the combined sequence.
    3) Write final deduplicated documents to a new JSONL folder.
    """
    pipeline_3 = [
        # Must be the same data that was passed to DatasetToSequence in stage 1
        JsonlReader("intermediate/"),

        # Removes duplicate segments based on the .bytearange file created by deduplicate-text-datasets
        ESRangeRemover(
            sequence_folder=f"{os.getcwd()}/es/",
        ),

        # Writes the cleaned data to a final folder
        JsonlWriter("final-deduped-data"),
    ]

    # Local executor to run the pipeline in parallel
    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, workers=4, tasks=4)

    print(executor_3.run())
