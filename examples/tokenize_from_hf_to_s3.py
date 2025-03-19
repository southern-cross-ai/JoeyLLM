"""
This script tokenizes a HuggingFace dataset in parallel using SLURM and then merges the tokenized outputs.
The process is split into two stages:
1. Distributed tokenization: Each task reads a portion of the dataset, tokenizes it, and saves the results.
2. Merging: The tokenized outputs are merged and shuffled into a final binary file for downstream training.

Command-line arguments allow you to specify the dataset, output locations, tokenizer, split, and SLURM settings.
"""

import argparse
import os.path

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

# -----------------------------------------------------------------------------
# Set up command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="huggingface dataset name. Example: stas/openwebtext-10k")
parser.add_argument(
    "output_path", type=str, help="Where to save individual tokenization files and the final merged files."
)
parser.add_argument("-l", "--logs", type=str, help="path to logs folder", default="tokenization_logs")
parser.add_argument("-t", "--tokenizer", type=str, help="tokenizer to use", default="gpt2")
parser.add_argument(
    "--local",
    type=str,
    help="local working directory. You won't need it if output_path is a local path. /scratch/datatrove/tokenize/ by default.",
    default="/scratch/datatrove/tokenize/",
)
parser.add_argument(
    "-o",
    "--output_name",
    type=str,
    help="filename for the final output files. By default this will be `dataset-tokenizer`",
    default=None,
)
parser.add_argument("-s", "--split", type=str, help="dataset split. `train` by default", default="train")
parser.add_argument(
    "-tk",
    "--text_key",
    type=str,
    help="Column that actually contains the text to be tokenized. `text` by default.",
    default="text",
)
parser.add_argument(
    "-p", "--partition", type=str, help="Slurm partition to use. `hopper-prod` by default.", default="hopper-prod"
)
parser.add_argument("-ts", "--tasks", type=int, help="Number of tasks to run. 1000 by default", default=1000)

# -----------------------------------------------------------------------------
# Setup for tokenization:
# - Reads a dataset from the HuggingFace hub.
# - Tokenizes the text using the specified tokenizer.
# - Saves the tokenized output to individual files and then merges them.
# -----------------------------------------------------------------------------

# Parse command-line arguments
args = parser.parse_args()

# Determine the final dataset name for output files.
DATASET_NAME = args.output_name  # Use provided output name if specified
if not DATASET_NAME:
    # Construct a default name from dataset and tokenizer, replacing "/" with "_"
    DATASET_NAME = f"{args.dataset}-{args.tokenizer}".replace("/", "_")

# Set up log and working directories.
LOGS_FOLDER = args.logs
# Directory where tokenized individual files will be saved.
WORKING_DIR = os.path.join(args.output_path, "tokenized-tasks")
# Local working directory if output path is not local; can be set to None if not needed.
LOCAL_WORKING_DIR = args.local
if LOCAL_WORKING_DIR:
    LOCAL_WORKING_DIR = os.path.join(LOCAL_WORKING_DIR, DATASET_NAME)
# Directory where the final merged tokenized dataset will be saved.
FINAL_OUTPUT_DIR = os.path.join(args.output_path, "merged-dataset")
# The FINAL_OUTPUT_DIR is the path that your training library will use to read the tokenized data.

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Stage 1: Distributed Tokenization
    # -----------------------------------------------------------------------------
    # This stage uses a SLURM pipeline executor to run multiple parallel tasks.
    # Each task reads a portion of the dataset, tokenizes the text, and writes the results.
    dist_executor = SlurmPipelineExecutor(
        job_name=f"{DATASET_NAME}-tok1",  # Unique job name for this tokenization stage.
        pipeline=[
            # Read the dataset from HuggingFace using the provided dataset name and options.
            HuggingFaceDatasetReader(
                dataset=args.dataset,  # Name of the dataset to load.
                dataset_options={
                    "split": args.split  # Specify the split (e.g., train, test).
                },
                text_key=args.text_key,  # Column in the dataset containing the text.
            ),
            # Tokenize documents using the specified tokenizer.
            DocumentTokenizer(
                output_folder=WORKING_DIR,  # Where to save individual tokenized files.
                local_working_dir=LOCAL_WORKING_DIR,  # Local working directory if needed.
                save_filename=f"{DATASET_NAME}_tokenized",  # Filename prefix for tokenized files.
                tokenizer_name_or_path=args.tokenizer,  # Name or path of the tokenizer (e.g., "gpt2").
            ),
        ],
        tasks=args.tasks,  # Number of parallel tasks for distributed tokenization.
        logging_dir=f"{LOGS_FOLDER}/tokenization",  # Directory to store SLURM logs for tokenization.
        time="20:00:00",  # Maximum allowed runtime for this stage.
        partition=args.partition,  # SLURM partition to use.
    )

    # -----------------------------------------------------------------------------
    # Stage 2: Merging Tokenized Outputs
    # -----------------------------------------------------------------------------
    # After distributed tokenization, this stage merges and shuffles all individual tokenized files
    # into a final tokenized binary file that can be used by training libraries.
    merge_executor = SlurmPipelineExecutor(
        job_name=f"{DATASET_NAME}-tok2",  # Unique job name for the merging stage.
        pipeline=[
            # Merge tokenized files from the working directory into a single dataset.
            DocumentTokenizerMerger(
                input_folder=WORKING_DIR,  # Folder containing individual tokenized outputs.
                output_folder=FINAL_OUTPUT_DIR,  # Folder where the final merged dataset will be saved.
                save_filename=f"{DATASET_NAME}",  # Filename for the merged output.
            ),
        ],
        tasks=1,  # Merging typically runs as a single task.
        logging_dir=f"{LOGS_FOLDER}/merged",  # Directory to store SLURM logs for merging.
        time="50:00:00",  # Maximum allowed runtime for the merging stage.
        partition=args.partition,  # SLURM partition to use.
        depends=dist_executor,  # Ensure that this stage only starts after tokenization is complete.
    )

    # Run the merging stage to produce the final tokenized dataset.
    merge_executor.run()

