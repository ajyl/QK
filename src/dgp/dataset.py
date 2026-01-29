from datasets import Dataset


class BindingDataset:
    """
    A Dataset class for studying binding.

    Attributes:
        id (str): Identifier for the dataset.
    """

    def __init__(self, dataset: Dataset, id="null"):
        """
        Initialize an instance.

        Args:
            dataset (Dataset).
            id (str, optional): Identifier for the dataset. Defaults to "null".
            **kwargs: Additional keyword arguments passed to the parent Dataset class
                     when creating a new dataset (if dataset is None).

        Raises:
            AssertionError: If required features "input" or "counterfactual_inputs"
                            are missing in the provided dataset.
        """
        self.id = id
        self.dataset = dataset

    @classmethod
    def from_dict(cls, data_dict, id="null"):
        """
        Create Dataset from a dictionary.

        Args:
            data_dict (dict): Dictionary containing "input" and "counterfactual_inputs".

        Returns:
            BindingDataset: A new BindingDataset instance.
        """
        dataset = Dataset.from_dict(data_dict)
        return cls(dataset=dataset, id=id)

    @classmethod
    def from_sampler(cls, sampler, size, id=None):
        """
        Generate a dataset of counterfactual examples.

        Creates a new dataset by repeatedly sampling inputs and their counterfactuals
        using the provided sampling function, optionally filtering the samples.

        Args:
            sampler (callable): Function that returns a dictionary
                                            with keys "input" and "counterfactual_inputs".
            size (int): Number of examples to generate.
        Returns:
            BindingDataset: A new CounterfactualDataset containing the generated examples.
        """
        inputs = []
        while len(inputs) < size:
            sample = sampler()
            inputs.append(sample)

        dataset = Dataset.from_dict(
            {
                "input": inputs,
                # "counterfactual_inputs": counterfactuals
            }
        )
        return cls(dataset=dataset, id=id)

    def display_data(self, num_examples=1, verbose=True):
        """
        Display examples from the dataset, showing both the original inputs
        and their corresponding counterfactual inputs.

        Args:
            num_examples (int, optional): Number of examples to display. Defaults to 1.
            verbose (bool, optional): Whether to print additional information such as
                                    dataset ID and formatting. Defaults to True.

        Returns:
            dict: A dictionary containing the displayed examples for programmatic access.
        """
        if verbose:
            print(f"Dataset '{self.id}':")

        displayed_examples = {}

        for i in range(min(num_examples, len(self))):
            example = self.dataset[i]

            if verbose:
                print(f"\nExample {i+1}:")
                print(f"Input: {example['input']}")
                # print(f"Counterfactual Inputs ({len(example['counterfactual_inputs'])} alternatives):")

                # for j, counterfactual_input in enumerate(example["counterfactual_inputs"]):
                #    print(f"  [{j+1}] {counterfactual_input}")

            # Store for programmatic access
            displayed_examples[i] = {
                "input": example["input"],
                # "counterfactual_inputs": example["counterfactual_inputs"]
            }

        if verbose and len(self) > num_examples:
            print(f"\n... {len(self) - num_examples} more examples not shown")

        return displayed_examples

    def add_column(self, column_name, column_data):
        """
        Add a new column to the dataset.

        Args:
            column_name (str): Name of the new column.
            column_data (list): Data for the new column.

        Raises:
            ValueError: If the length of column_data does not match the number of examples in the dataset.
        """
        if len(column_data) != len(self.dataset):
            raise ValueError(
                f"Length of {column_name} must match number of examples in dataset."
            )

        self.dataset = self.dataset.add_column(column_name, column_data)

    def remove_column(self, column_name):
        """
        Remove a column from the dataset.

        Args:
            column_name (str): Name of the column to remove.
        """
        self.dataset = self.dataset.remove_columns(column_name)

    def __getitem__(self, idx):
        """
        Get an example from the dataset by index.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            dict: The example at the specified index, containing "input" and
                    "counterfactual_inputs".
        """
        return self.dataset[idx]

    def __len__(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.dataset)
