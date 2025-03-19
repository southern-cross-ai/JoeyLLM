from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

# Define the dataset name to be processed.
DATASET_NAME = "c4"

# Pipeline 1: Tokenization
# This pipeline reads the C4 dataset directly from Hugging Face, filters for English files,
# and tokenizes the documents. The tokenized output is saved to S3 and a local scratch directory.
dist_executor = SlurmPipelineExecutor(
    job_name="c4_tok_1",
    pipeline=[
        JsonlReader(
            "hf://datasets/allenai/c4/en/",  # Read from Hugging Face
            glob_pattern="c4-*.json.gz",       # Only process English files
        ),
        DocumentTokenizer(
            output_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
            local_working_dir=f"/scratch/guilherme/{DATASET_NAME}/tokenized/",
            save_filename=f"{DATASET_NAME}_tokenized",
        ),
    ],
    tasks=1001,              # Number of parallel tasks for tokenization
    workers=64,              # Number of workers per task
    time="72:00:00",         # Maximum runtime for the tokenization stage
    partition="production-cluster",
    logging_dir=f"/fsx/guilherme/logs/tokenize_{DATASET_NAME}",
)
dist_executor.run()

# Pipeline 2: Merge Tokenized Outputs
# This pipeline merges the individual tokenized outputs into a standardized dataset.
# It depends on the successful completion of the tokenization stage.
merge_executor = SlurmPipelineExecutor(
    job_name="c4_tok_2",
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
            output_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/standard/",
            save_filename=f"{DATASET_NAME}",
        ),
    ],
    tasks=1,
    time="50:00:00",
    partition="production-cluster",
    logging_dir=f"/fsx/guilherme/logs/tokenize_{DATASET_NAME}_merged",
    mem_per_cpu_gb=11,
    depends=dist_executor,  # Wait for the tokenization pipeline to finish before merging
)
merge_executor.run()
