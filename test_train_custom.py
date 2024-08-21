# Assemble datasets for model training and performance evaluation
# Before running this script, run `training_extract_audio_examples.py` to extract audio examples to 'data/training/Custom'
# TODO: REPLACE THIS WITH train.py in the root directory of the src repo

training_data_path = 'data/training/Custom'
output_path = 'data/models/custom'
test_set_size = 25
development_set_size = 125

preexisting_labels_to_train = [
    "american robin",
    "band-tailed pigeon",
    "barred owl",
    "belted kingfisher",
    "black-throated gray warbler",
    "common raven",
    "dark-eyed junco",
    "golden-crowned kinglet",
    "hairy woodpecker",
    "hammond's flycatcher",
    "hermit thrush",
    "hutton's vireo",
    "marbled murrelet",
    "northern flicker",
    "northern pygmy-owl",
    "northern saw-whet owl",
    "olive-sided flycatcher",
    "pacific wren",
    "pacific-slope flycatcher",
    "pileated woodpecker",
    "purple finch",
    "red crossbill",
    "red-breasted nuthatch",
    "ruby-crowned kinglet",
    "rufous hummingbird",
    "song sparrow",
    "sooty grouse",
    "spotted towhee",
    "swainson's thrush",
    "townsend's warbler",
    "varied thrush",
    "violet-green swallow",
    "western screech-owl",
    "western tanager",
    "western wood-pewee",
    "white-crowned sparrow",
    "wilson's warbler"
]
novel_labels_to_train = [
    "abiotic aircraft",
    "abiotic logging",
    "abiotic rain",
    "abiotic vehicle",
    "abiotic wind",
    "biotic anuran",
    "biotic insect"
]
labels_to_train = preexisting_labels_to_train + novel_labels_to_train

import random
import pandas as pd
import numpy as np
import os
import sys
import subprocess

# Prepare output directory
os.makedirs(output_path, exist_ok=True)

# Find all selection table files under a root directory
def find_files(directory, suffix=None, prefix=None):

    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            suffix_match = (suffix is None) or (suffix is not None and file.endswith(suffix))
            prefix_match = (prefix is None) or (prefix is not None and file.startswith(prefix))
            if suffix_match and prefix_match:
                results.append(os.path.join(root, file))
    return results

# Normalize file paths to support both mac and windows
training_data_path = os.path.normpath(training_data_path)

# Find all potential training files for all labels
training_data_audio_path = training_data_path + '/audio'
print(f'Finding all audio data for model development from {training_data_audio_path}...')
training_filepaths = []
training_filepaths.extend(find_files(training_data_audio_path, '.wav'))
available_examples = pd.DataFrame()
for path in training_filepaths:
    dir   = os.path.basename(os.path.dirname(path))
    label_tokens = dir.split('_')
    label = label_tokens[1].lower() if len(label_tokens) == 2 else ''
    if label not in labels_to_train:
        continue
    example = pd.DataFrame({
        'label':  [label],
        'folder': [os.path.basename(os.path.dirname(path))],
        'file':   [os.path.basename(path)],
        'path':   [path]
    })
    available_examples = pd.concat([available_examples, example], ignore_index=True)

# # Find all training annotations and associated audio data for all labels
# training_data_selections_path = training_data_path + '/selections'
# print(f'Finding all training annotation data and associated audio data from {training_data_selections_path}...')
# annotation_files = []
# annotation_files.extend(files.find_files(training_data_selections_path, '.txt')) 
# all_annotations = pd.DataFrame()
# for file in annotation_files:
#     example = files.load_raven_selection_table(file, cols_needed = ['label', 'file_audio'])
#     example['label_source'] = os.path.basename(os.path.dirname(file))
#     all_annotations = pd.concat([all_annotations, example], ignore_index=True)
# print(all_annotations.head().to_string())

# training_annotations = all_annotations

# DO NOT CHANGE
# Never change this random seed to ensure unbiased evaluation of test data.
test_seed = 1

# Randomly set aside 25 examples of each novel label as test data
print(f'Randomly choosing {test_set_size} examples for each novel label (N={len(novel_labels_to_train)}) as test data...')
test_examples_novel = pd.DataFrame()
for novel_label in novel_labels_to_train:
    label_examples = available_examples[available_examples['label'] == novel_label]
    # Randomly sample 25 examples
    label_sample = label_examples.sample(n=test_set_size, random_state=test_seed)
    test_examples_novel = pd.concat([test_examples_novel, label_sample], ignore_index=True)
    # Remove the samples from the training data
    available_examples = available_examples.drop(label_sample.index)
print(test_examples_novel['label'].value_counts())

# Store test example filepaths
test_files_csv_path = f'{output_path}/test_files.csv'
test_examples_novel.to_csv(test_files_csv_path, index=False)

# ========================================================================================================================================================================================================
# TRAIN WITH NO CROSS VALIDATION

# Set a random seed
training_seed = 1

# Create a development (train + validation) dataset by randomly choosing 125 examples (or, if less than 125 exist, as many as are available) from the total available for each label.
print(f'Randomly choosing {development_set_size} examples for each label as development data...')
development_examples = pd.DataFrame()
for label_to_train in labels_to_train:
    print(label_to_train)
    label_examples = available_examples[available_examples['label'] == label_to_train]
    # Randomly sample 125 examples
    if len(label_examples) < development_set_size:
        print(f'WARNING: Less than {development_set_size} examples available for label {label_to_train}')
    label_sample = label_examples.sample(n=min(development_set_size, len(label_examples)), random_state=training_seed)
    development_examples = pd.concat([development_examples, label_sample], ignore_index=True)
print(f'Found {len(development_examples)} total development examples')
print(development_examples['label'].value_counts())

# TODO: For labels with fewer than 125 examples, apply data augmentation to artificially increase the number of examples to 125 (e.g. SMOTE).

# Split development data into 20% validation data (25 examples for labels with 125 total) and 80% training data (100 examples for labels with 125 total).
# This validation data will be used across all model experiments
training_set_proportion = 0.80
validation_set_proportion = round(1.0 - training_set_proportion, 2)
print(f'Randomly choosing {training_set_proportion * 100}/{validation_set_proportion * 100}% for each label as training/validation data...')
validation_examples = pd.DataFrame()
training_examples   = pd.DataFrame()
for label, group in development_examples.groupby('label'):
    label_validation = group.sample(frac=validation_set_proportion, random_state=training_seed)
    label_training   = group.drop(label_validation.index)
    validation_examples = pd.concat([validation_examples, label_validation]).reset_index(drop=True)
    training_examples   = pd.concat([training_examples, label_training]).reset_index(drop=True)
print(validation_examples['label'].value_counts())
print(training_examples['label'].value_counts())

# Store validation example filepaths
validation_files_csv_path = f'{output_path}/validation_files.csv'
validation_examples.to_csv(validation_files_csv_path, index=False)

# Define sample size experiments for model development
sample_size_experiments = [2]

# For each experiment (2,5,10,25,50,100 training examples)...
for experiment_sample_size in sample_size_experiments:
    print(f'Running model development with {experiment_sample_size} training samples...')

    autotune = 0 # TODO: run another loop with autotune
    model_id_stub = f'custom_S{training_seed}_N{experiment_sample_size}_A{autotune}'
    path_model_out = f'{os.getcwd()}/{output_path}/{model_id_stub}'
    file_model_out = f'{path_model_out}/{model_id_stub}.tflite'
    os.makedirs(path_model_out, exist_ok=True)

    # Randomly sample the required number of examples per label from the training data
    experiment_examples = pd.DataFrame()
    for label, group in training_examples.groupby('label'):
        label_examples = training_examples[training_examples['label'] == label]
        if len(label_examples) < experiment_sample_size:
            print(f'WARNING: Less than {experiment_sample_size} examples available for label {label}')
            # TODO: Apply data augmentation to artificially increase the number of examples to 125 (e.g. SMOTE)?
        label_training = group.sample(n=min(experiment_sample_size, len(label_examples)), random_state=training_seed)
        experiment_examples = pd.concat([experiment_examples, label_training]).reset_index(drop=True)
    print(experiment_examples['label'].value_counts())
    experiment_examples = experiment_examples.sort_values(by=['path'])
    print(experiment_examples.to_string())

    # Store training filepaths
    training_files_csv_path = f'{output_path}/{model_id_stub}/training_files.csv'
    experiment_examples.to_csv(training_files_csv_path, index=False)

    # Train the model on these samples 
    print(f'Training model {os.path.basename(file_model_out)} with {experiment_sample_size} examples ======================================================================================')
    args = [
        'python3', 'src/submodules/BirdNET-Analyzer/train_custom.py',
        '--i', '/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/training/Custom/audio', # Path to training data folder.
        '--o', file_model_out, # File path to trained classifier model output.
        '--no-autotune' if not autotune else '--autotune' # Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).
    ]
    print(args)

    # Run the script with arguments
    sys.argv = args[1:]
    with open(args[1]) as path:
        exec(path.read())

    print('done')
    
    # Evaluate model performance with the shared validation data
    # TODO


# ========================================================================================================================================================================================================
# TRAIN WITH 5-FOLD CROSS VALIDATION
# - Set a random seed
# - Create a development (train + validation) dataset by randomly choosing 125 examples (or, if less than 125 exist, as many as are available) from the total available for each label.
# - For labels with fewer than 125 examples, choose all available examples for training, then apply data augmentation to artificially increase the number of examples to 125 (e.g. SMOTE).
# - Split this 125 into 5 randomly-chosen folds of validation data (25 examples for labels with 125 total) and training data (100 examples for labels with 125 total) with a 80/20 split, each fold being of size 25:
# 	V T T T T
# 	T V T T T
# 	T T V T T
# 	T T T V T
# 	T T T T V
# - For each split...
# 	- Get the training folds
# 	- Get the validation fold
# 	- For each condition (2,5,10,25,50,100)...
# 		- Randomly sample the required number of examples per label from the training folds for the split
# 		- Train the model on this sample (both without and with hyperparameter autotune)
# 		- Evaluate model performance with the validation fold for the split
# - For each condition (5,10,25,50,100)...
# 	- Average the model's performance across all 5 splits for the condition