import argparse

# Set up command-line arguments for input dataset, output name, number of tasks, and text column.
parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")
parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("output_name", type=str, help="Name of the output dataset")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=100)
parser.add_argument("--text_key", type=str, help="text column", default="text")

# Global configuration for organization and local paths.
ORG_NAME = "my_org"
LOCAL_PATH = "my_local_path"
LOCAL_LOGS_PATH = "my_local_logs_path"

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Import key modules from datatrove for building and executing the pipeline.
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter

    # Create a SLURM pipeline executor to distribute processing across tasks.
    # The pipeline consists of:
    # 1. Reading Parquet files from the input dataset.
    # 2. Filtering documents to include only those with "hugging" in the text.
    # 3. Writing the filtered results back to the Hugging Face hub.
    dist_executor = SlurmPipelineExecutor(
        job_name=f"filter-{args.output_name}",
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="**/*.parquet", text_key=args.text_key),
            LambdaFilter(lambda doc: "hugging" in doc.text),
            HuggingFaceDatasetWriter(
                dataset=f"{ORG_NAME}/{args.output_name}",
                private=True,
                local_working_dir=f"{LOCAL_PATH}/{args.output_name}",
                output_filename="data/${rank}.parquet",
                cleanup=True,
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=f"{LOCAL_LOGS_PATH}/{args.output_name}",
        cpus_per_task=12,
        qos="high",
        mem_per_cpu_gb=3,
    )

    # Run the distributed pipeline.
    dist_executor.run()
