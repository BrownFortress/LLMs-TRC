# Encoder Architecture


In this folder, you can find the scripts of the encoder architecture. The folder is organized as follows:

- `bin`: the place where the model's weights are saved.

- `configs`: hyperparamter and configs of the experiments.

- `data`: the data used to train the model. You have to preprocess the data using the scripts provided in `data_formatter` and copy&paste the output here.

- `frozen_embeddings`: the place where pre-computed embeddings are stored.
- `output`: it stores the files containing the results of the experiments.
- `sandbox`: classes used to define the data structures used for training and testing the models.

- `utils`: classes that define the encoder architecture and the corresponding data loader. Functions used to train and test the models.

- __main.py__: script for running the experiments.

- __print_tab.py__, __stat_reader.py__ and __stats.py__: scripts for printing the results in a human-readable format.

To re-run the experiments, you can look at the file __run_exps.sh__.