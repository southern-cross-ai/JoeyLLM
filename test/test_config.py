# File path: test/test_config.py

import sys
import os



from src.configs.config import Config

import unittest
import os
import yaml


class TestConfig(unittest.TestCase):

    def setUp(self):
        # Set up the path to the YAML configuration file. 
        # Assumes tests are run from the root of the repository.
        self.config_path = "src/configs/config.yaml"

    def test_yaml_loading(self):
        """
        Test if the YAML file can be loaded successfully.
        """
        with open(self.config_path, "r") as file:
            try:
                # Load the YAML file and ensure it is parsed into a dictionary.
                raw_data = yaml.safe_load(file)
                self.assertIsInstance(raw_data, dict, "The YAML file should load as a dictionary.")
            except yaml.YAMLError as e:
                # Fail the test if the YAML file cannot be loaded.
                self.fail(f"Failed to load the YAML file: {e}")

    def test_config_validation(self):
        """
        Test if the YAML configuration adheres to the schema defined in the Config model.
        """
        # Use the `from_yaml` method to load and validate the configuration.
        config = Config.from_yaml(self.config_path)
        self.assertIsInstance(config, Config, "The loaded configuration should be an instance of the Config class.")
        # Ensure that vocab_size is a positive integer.
        self.assertGreater(config.vocab_size, 0, "vocab_size should be greater than 0.")

    def test_config_data_types(self):
        """
        Test if the data in the YAML file matches the expected data types.
        """
        with open(self.config_path, "r") as file:
            # Load the raw YAML data.
            raw_data = yaml.safe_load(file)
        
        # Validate the 'model' section of the configuration.
        model = raw_data.get("model", {})
        self.assertIsInstance(model.get("type"), str, "model.type should be a string.")
        self.assertIsInstance(model.get("vocab_size"), int, "model.vocab_size should be an integer.")
        self.assertIsInstance(model.get("max_seq_len"), int, "model.max_seq_len should be an integer.")
        self.assertIsInstance(model.get("embed_dim"), int, "model.embed_dim should be an integer.")
        self.assertIsInstance(model.get("num_heads"), int, "model.num_heads should be an integer.")
        self.assertIsInstance(model.get("num_layers"), int, "model.num_layers should be an integer.")
        self.assertIsInstance(model.get("dropout"), float, "model.dropout should be a float.")

        # Validate the 'data' section of the configuration.
        data = raw_data.get("data", {})
        self.assertIsInstance(data.get("dataset_out"), str, "data.dataset_out should be a string.")
        self.assertIsInstance(data.get("dataset_in"), str, "data.dataset_in should be a string.")
        self.assertIsInstance(data.get("batch_size"), int, "data.batch_size should be an integer.")
        self.assertIsInstance(data.get("columns"), list, "data.columns should be a list.")
        self.assertIsInstance(data.get("format"), str, "data.format should be a string.")
        self.assertIsInstance(data.get("shuffle"), bool, "data.shuffle should be a boolean.")
        self.assertIsInstance(data.get("use_validation"), bool, "data.use_validation should be a boolean.")
        self.assertIsInstance(data.get("use_test"), bool, "data.use_test should be a boolean.")

        # Validate the 'train' section of the configuration.
        train = raw_data.get("train", {})
        self.assertIsInstance(train.get("batch_size"), int, "train.batch_size should be an integer.")
        self.assertIsInstance(train.get("device"), str, "train.device should be a string.")
        self.assertIsInstance(train.get("epochs"), int, "train.epochs should be an integer.")
        self.assertIsInstance(train.get("learning_rate"), float, "train.learning_rate should be a float.")
        self.assertIsInstance(train.get("weight_decay"), float, "train.weight_decay should be a float.")
        self.assertIsInstance(train.get("save_every"), int, "train.save_every should be an integer.")
        # 'resume_from' can be either None or a string.
        self.assertTrue(train.get("resume_from") is None or isinstance(train.get("resume_from"), str),
                        "train.resume_from should be a string or None.")
        self.assertIsInstance(train.get("gradient_accumulation_steps"), int, "train.gradient_accumulation_steps should be an integer.")
        self.assertIsInstance(train.get("checkpoint_path"), str, "train.checkpoint_path should be a string.")
        # Validate the 'wandb' subsection.
        wandb = train.get("wandb", {})
        self.assertIsInstance(wandb.get("project"), str, "train.wandb.project should be a string.")
        self.assertIsInstance(wandb.get("entity"), str, "train.wandb.entity should be a string.")
        self.assertIsInstance(wandb.get("log"), bool, "train.wandb.log should be a boolean.")

if __name__ == "__main__":
    unittest.main()