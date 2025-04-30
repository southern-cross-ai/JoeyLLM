import argparse
import dataclasses

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.stats import DocStats, LineStats, StatsMerger, TopKConfig, WordStats

# Total number of parallel tasks to use for the SLURM jobs.
TOTAL_TASKS = 500

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Summary Stats")
parser.add_argument("dump_path", help="Dump name sampler")  # Name/path of the dump to process
parser.add_argument("sample_rate", type=float, help="Sample rate")  # Sampling rate to reduce the dataset
parser.add_argument("--prefix", default="", help="Prefix")  # Optional prefix to prepend to the dump path
parser.add_argument("--glob", help="Glob pattern")  # Optional glob pattern to filter files
parser.add_argument("--text_key", default="text", help="Text key")  # Key used in the JSON objects for text content
parser.add_argument("--reader", default="jsonl", help="Reader type")  # Type of reader to use (default is JSONL)

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    # Generate a unique experiment name by replacing any "/" in the dump path with "_"
    experiment_name = args.dump_path.replace("/", "_")

    # Define local logging directory and S3 data folder based on the experiment name
    LOCAL_LOGS_FOLDER = f"/logs/{experiment_name}"
    DATA_FOLDER = f"s3://data/{experiment_name}"

    # Construct the source path for reading input data
    SOURCE = f"{args.prefix}/{args.dump_path}"
    print(SOURCE)

    # Configure the top-k statistics: tracking the top 10,000 items per group (e.g., fqdn, suffix)
    top_k_config = TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=10_000)

    # -----------------------------------------------------------------------------
    # Define the compute pipeline to generate summary statistics
    # -----------------------------------------------------------------------------
    compute = SlurmPipelineExecutor(
        pipeline=[
            # Read JSONL input files from the SOURCE.
            # 'doc_progress=True' enables progress reporting,
            # 'limit=-1' implies no limit on the number of documents, and 'glob_pattern' filters the files.
            JsonlReader(SOURCE, doc_progress=True, limit=-1, glob_pattern=args.glob, text_key=args.text_key),
            
            # Apply a sampling filter to reduce the dataset based on the provided sample rate.
            SamplerFilter(
                rate=args.sample_rate,
            ),
            
            # Compute word-level statistics (e.g., word frequency, top-k words).
            WordStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
            
            # Compute line-level statistics (e.g., average line length).
            LineStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
            
            # Compute document-level statistics (e.g., total documents, document lengths).
            DocStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
        ],
        tasks=TOTAL_TASKS,  # Number of parallel tasks for compute stage.
        job_name=f"summary-stats-{experiment_name}",  # Unique job name for SLURM.
        time="24:00:00",  # Maximum runtime for the compute job.
        partition="hopper-cpu",  # SLURM partition to run the job.
        logging_dir=f"{LOCAL_LOGS_FOLDER}-compute",  # Directory for compute stage logs.
        qos="normal",
        mem_per_cpu_gb=2,
        cpus_per_task=1,
    )

    # -----------------------------------------------------------------------------
    # Define the merger pipeline to merge statistics from all tasks
    # -----------------------------------------------------------------------------
    merger = SlurmPipelineExecutor(
        pipeline=[
            # Merge all statistics from the compute stage.
            # 'remove_input=False' ensures that the input data is not deleted after merging.
            # Here we update the top-k configuration for the merging stage (top_k=8,000).
            StatsMerger(
                input_folder=DATA_FOLDER,
                output_folder=f"{DATA_FOLDER}",
                remove_input=False,
                top_k_config=dataclasses.replace(top_k_config, top_k=8_000),
            ),
        ],
        tasks=TOTAL_TASKS,  # Number of parallel tasks for merging.
        job_name=f"merging-stats-{experiment_name}",  # Unique job name for the merging stage.
        time="24:00:00",  # Maximum runtime for the merge job.
        partition="hopper-cpu",
        logging_dir=f"{LOCAL_LOGS_FOLDER}-merge",  # Directory for merger logs.
        qos="normal",
        mem_per_cpu_gb=2,
        cpus_per_task=1,
        depends=compute,  # This job will start only after the compute pipeline has completed.
    )

    # Run the merger pipeline to combine statistics
    merger.run()
