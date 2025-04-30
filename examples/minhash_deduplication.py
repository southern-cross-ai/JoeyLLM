from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages

# Configure Minhash settings.
# Higher precision (64 bits) reduces false positives by improving hash accuracy.
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)

# Define global paths and task parameters.
S3_MINHASH_BASE_PATH = "s3://mybucket/minhash/"
S3_LOGS_FOLDER = "s3://mybucket/my_minhash_logs_path/"
LOCAL_LOGS_FOLDER = "my_local_folder_for_slurm_logs/"
TOTAL_TASKS = 1000

# Define the reader for the original JSONL data stored on S3.
INPUT_READER = JsonlReader("s3://mybucket/base_processing/output/")

# --- Stage 1: Signature Generation ---
# Compute minhash signatures for each document to prepare for deduplication.
stage1 = SlurmPipelineExecutor(
    job_name="mh1",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
            config=minhash_config,
            language=Languages.english
        ),
    ],
    tasks=TOTAL_TASKS,
    time="5:00:00",
    partition="hopper-cpu",
    logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
    qos="high",
)

# --- Stage 2: Bucketing ---
# Group similar signatures into buckets to identify potential duplicate pairs.
stage2 = SlurmPipelineExecutor(
    job_name="mh2",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
            output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    time="90:00:00",
    partition="hopper-prod",
    logging_dir=f"{S3_LOGS_FOLDER}/buckets",
    depends=stage1,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/slurm_logs",
    qos="high",
)

# --- Stage 3: Clustering ---
# Cluster documents into duplicate groups based on bucket matches.
stage3 = SlurmPipelineExecutor(
    job_name="mh3",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,
    time="90:00:00",
    partition="hopper-prod",
    logging_dir=f"{S3_LOGS_FOLDER}/clusters",
    mem_per_cpu_gb=70,
    cpus_per_task=2,
    depends=stage2,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clusters/slurm_logs",
)

# --- Stage 4: Filtering and Writing ---
# Read the original data, count tokens (for monitoring), and remove duplicates
# (keeping one sample per duplicate cluster), then write the clean data back.
stage4 = SlurmPipelineExecutor(
    job_name="mh4",
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # Helps compare token counts pre- and post-deduplication.
        MinhashDedupFilter(
            input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            exclusion_writer=JsonlWriter(f"{S3_MINHASH_BASE_PATH}/removed"),
        ),
        JsonlWriter(output_folder=f"{S3_MINHASH_BASE_PATH}/deduplicated_output"),
    ],
    tasks=TOTAL_TASKS,
    time="50:00:00",
    partition="hopper-cpu",
    logging_dir=f"{S3_LOGS_FOLDER}/filter",
    depends=stage3,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filter/slurm_logs",
)

# Run the final stage to trigger the entire deduplication pipeline.
stage4.run()
