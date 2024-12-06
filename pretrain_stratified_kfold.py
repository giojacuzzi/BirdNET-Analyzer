# Assemble datasets for model training and validation
#
# Input:
# - Path to directory containing all training audio files
# - Table containing all training data annotations
#
# Output:
# - Table of development file references for training and validation datasets (data/figures/interim/{model_config_stub}) 
#
# User-defined parameters:
training_data_audio_path = 'data/training/audio'
training_data_annotations_path = 'data/training/training_data_annotations.csv'
target_species_list_file = 'models/target/target_species_list.txt'

# Core hyperparameters
k = 5                           # Number of folds (1, i.e. no folds, for final model)
sample_sizes = [10]             # Sample size(s) for model development, e.g. [2, 5, 10, 25, 50, 75, 100] ([125] for final model).
epochs = 50                     # Default 50 (15 for final model, average of epochs across xval model's best validation losses).
learning_rate = 0.001           # Default 0.001 (0.001 for final model)
batch_size    = 10              # Default N (10 for final model)
hidden_units  = 0               # Default 0 (0 for final model)
training_set_proportion = 0.80  # Remainder reserved for validation (1.0 for final model, i.e. no validation)
development_set_size = 125      # Total training + validation
test_set_size = 25              # For novel labels only (25)

# Other hyperparameters
label_smooth  = False # Default False
upsample      = False
############################

import random
import pandas as pd
import os
import copy

if __name__ == "__main__":

    # Set random seeds
    training_seed = 1 
    random.seed(training_seed)
    test_seed = 1 # NOTE: This test seed must remain constant to ensure unbiased evaluation of held-out test data for novel labels

    target_model_stub = f'custom_S{training_seed}_LR{learning_rate}_BS{batch_size}_HU{hidden_units}_LS{label_smooth}_US{int(upsample)}'
    output_path = f'data/interim/{target_model_stub}/training'

    # Load class labels
    source_class_labels = pd.read_csv(os.path.abspath(f'models/source/source_species_list.txt'), header=None)[0].tolist()
    target_class_labels = pd.read_csv(os.path.abspath(f'models/target/target_species_list.txt'), header=None)[0].tolist()

    novel_labels_to_train = [l for l in target_class_labels if l not in source_class_labels]
    labels_to_train = target_class_labels

    print(f"{len(novel_labels_to_train)} novel labels to train:")
    print(novel_labels_to_train)
    print()
    print(f"{len(labels_to_train)} total labels to train:")
    print(labels_to_train)
    print()

    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)

    # Find all files under a root directory
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
    training_data_path = os.path.normpath(training_data_audio_path)

    # Find all potential training files for all labels
    print(f'Finding all audio data for model development from {training_data_audio_path}...')
    training_filepaths = []
    training_filepaths.extend(find_files(training_data_audio_path, '.wav'))

    available_examples = pd.DataFrame()
    for path in training_filepaths:
        dir = os.path.basename(os.path.dirname(path))
        example = pd.DataFrame({
            'audio_subdir': [os.path.basename(os.path.dirname(path))],
            'file':   [os.path.basename(path)],
            'path':   [os.path.abspath(path)]
        })
        available_examples = pd.concat([available_examples, example], ignore_index=True)

    # Load annotation labels for the training files and merge, retaining only those that match the available examples
    annotations = pd.read_csv(training_data_annotations_path)
    annotations['file'] = annotations['file'] + '.wav'
    available_examples = available_examples.merge(annotations[['audio_subdir', 'file', 'labels']], on=['audio_subdir', 'file'], how='left')
    available_examples.loc[available_examples['audio_subdir'] == 'Background', 'labels'] = 'Background' # manually label background examples

    # Randomly set aside examples of each novel label as test data
    print(f'Randomly choosing {test_set_size} examples for each novel label (N={len(novel_labels_to_train)}) as test data...')
    test_examples_novel = pd.DataFrame()
    for novel_label in novel_labels_to_train:
        if novel_label == 'Background': # skip the Background label
            continue
        label_examples = available_examples[available_examples['audio_subdir'] == novel_label]
        sampled_rows = label_examples.sample(n=test_set_size, random_state=test_seed)
        test_examples_novel = pd.concat([test_examples_novel, sampled_rows], ignore_index=True)
        available_examples = available_examples.drop(sampled_rows.index) # Remove the samples from the training data

    # Store test example filepaths
    os.makedirs(f'data/test/{target_model_stub}', exist_ok=True)
    novel_test_files_csv_path = os.path.abspath(f'data/test/{target_model_stub}/novel_test_files.csv')
    test_examples_novel.to_csv(novel_test_files_csv_path, index=False)
    print(f'Saved test example filepaths to {novel_test_files_csv_path}')

    # Find the majority and minority classes (i.e. value_couts across all labels present in available_examples)
    def class_imbalance_test(df_in, print_out=False):
        if df_in.empty:
            print('WARNING: Unable to calculate class imbalance with empty dataframe')
            return
        if print_out:
            print('Finding majority and minority classes from label value counts...')
        df = copy.deepcopy(df_in)
        df['labels'] = df['labels'].str.split(', ')
        df = df.explode('labels')
        label_counts = df['labels'].value_counts()
        label_counts = label_counts[label_counts.index.isin(labels_to_train + ['Background'])]
        max_count = label_counts.max()
        min_count = label_counts.min()
        max_labels = label_counts[label_counts == max_count].index.tolist()
        min_labels = label_counts[label_counts == min_count].index.tolist()
        if print_out:
            print(label_counts.to_string())
            print(f"Majority classes  (N={max_count}): {', '.join(max_labels)}")
            print(f"Miniority classes (N={min_count}): {', '.join(min_labels)}")
            print(f"Class imbalance ratio ({max_count}/{min_count}): {round(max_count/min_count,2)}")
        return(label_counts)

    # Create a development (train + validation) dataset by randomly choosing 125 examples (or, if less than 125 exist, as many as are available) from the total available for each label.
    print(f'Randomly choosing {development_set_size} examples for each label as development data...')
    development_examples = pd.DataFrame()
    for label_to_train in (labels_to_train + ['Background']):
        label_examples = available_examples[available_examples['labels'].str.contains(label_to_train, regex=False)]
        if len(label_examples) < development_set_size:
            print(f'WARNING: Less than {development_set_size} examples available for label {label_to_train}')
        sampled_rows = label_examples.sample(n=min(development_set_size, len(label_examples)), random_state=training_seed)
        development_examples = pd.concat([development_examples, sampled_rows], ignore_index=True)
        available_examples = available_examples.drop(sampled_rows.index) # Remove the sampled examples from 'available_examples'
    print(f'Found {len(development_examples)} total development examples')

    print('Development example count by class:')
    class_counts = class_imbalance_test(development_examples, print_out=True)

    # ========================================================================================================================================================================================================
    # PREPARE TRAIN WITH STRATIFIED K-FOLD CROSS VALIDATION
    print(f'Preparing data for training (stratified k-fold cross-validation) ======================================================================================')

    labels = (labels_to_train + ['Background'])

    def evenly_distribute_n_into_parts(n, parts):
        quotient, remainder = divmod(n, parts)
        result = [quotient] * parts
        for i in range(remainder): result[i] += 1
        return result

    # Perform iterative stratification http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
    # Split the development dataset into k roughly equally-sized folds, ensuring that each fold has approximately the same class distribution as the original dataset
    # The splitting is done in such a way that each fold preserves the percentage of samples for each class as observed in the original dataset.
    # Input: A set of instances, D, annotated with a set of labels L = {λ1, ..., λq}, desired number of subsets k, desired proportion of examples in each subset, r1, . . . rk (e.g. in 10-fold CV k = 10, rj = 0.1, j = 1...10)
    def iterative_stratification(d, k):

        d = development_examples
        d['L'] = d['labels'].str.split(", ")

        # Calculate the desired number of examples cj at each subset (fold) Sj by multiplying the number of examples, |D|, with the desired proportion for this subset (fold) rj
        c = evenly_distribute_n_into_parts(len(d), k)
        print(f'Desired number of total samples for each fold: {c}')
        
        # Calculate the desired number of examples of each label λl at each subset (fold) Sj, clj, by multiplying the number of examples annotated with that label, |Dl|, with the desired proportion for this subset rj (i.e. according to the proportion of examples of the label in the initial set).  Note that both cj and clj will most often be decimal numbers.
        cl = pd.DataFrame(columns=['label', 'clj'])
        for l in labels:
            dl = d[d['L'].apply(lambda x: l in x)] # get all examples of the label
            clj = evenly_distribute_n_into_parts(len(dl), k)
            cl = pd.concat([cl, pd.DataFrame({'label': [l], 'clj': [clj]})], axis=0)
        print('Desired number of samples for each label:')
        print(cl.to_string())

        # The algorithm will finish as soon as the original dataset is empty
        d['fold'] = -1 # Initialize all examples to be unassigned to a fold
        while (-1 in d['fold'].values):
            # Find the label with the fewest (but at least one) remaining examples, breaking ties randomly. It examines one label in each iteration, the one with the fewest remaining examples, denoted l.
            label_counts = class_imbalance_test(d[d['fold'] == -1])
            print(label_counts)
            min_count_labels = label_counts[label_counts == label_counts.min()].index.tolist()
            l = random.choice(min_count_labels)
            print(f'Distributing: {l}')

            # Then, for each example (x, Y ) of this label dl, the algorithm selects an appropriate subset (fold) for distribution. The first criterion for subset (fold) selection is the current desired number of examples for this label clj. The subset (fold) that maximizes it gets selected.
            dl = d[d['L'].apply(lambda x: l in x)] # Retrieve all examples of the label
            dl = dl[dl['fold'] == -1] # Consider only examples that have yet to be assigned to a fold
            for index, example in dl.iterrows():

                # Select a fold for distribution of the sample, prioritizing the fold with the maximum number of remaining desired examples of this label
                clj = cl[cl['label'] == l]['clj'].iloc[0]
                max_clj_idx = [i for i, v in enumerate(clj) if v == max(clj)]
                
                # In case of ties, among the tying subsets (folds) select the one with the highest number of desired examples cj. Further ties are broken randomly.
                if len(max_clj_idx) > 1:
                    # Among the tying subsets (folds), the one with the highest number of desired examples cj is selected.
                    max_c_idx = [i for i, v in enumerate(c) if v == max(c)]
                    if len(max_c_idx) > 1:
                        m = random.choice(max_clj_idx)
                    else:
                        m = max_c_idx[0]

                else:
                    m = max_clj_idx[0]
                
                # If the selected fold no longer desires more examples overall (although it may desire more of this specific label),
                # choose the fold that desires the most examples overall instead
                if c[m] == 0:
                    max_c_idx = [i for i, v in enumerate(c) if v == max(c)]
                    if len(max_c_idx) > 1:
                        m = random.choice(max_c_idx)
                    else:
                        m = max_c_idx[0]

                # Once the appropriate subset, m, is selected, add the example (x, Y) to Sm and remove it from D.
                d.loc[index, 'fold'] = m

                # At the end of the iteration, decrement the number of desired examples for each label of this example at subset m, cim, as well as the total number of desired examples for subset m, cm
                cl[cl['label'] == l]['clj'].iloc[0][m] -= 1
                c[m] -= 1

            print(d[d['L'].apply(lambda x: l in x)]['fold'].value_counts())
            print(f'Remaining desired number of total samples for each fold: {c}')

        return(d)

    print(f'Beginning iterative stratification of development examples (k={k})...')
    d = iterative_stratification(d=development_examples, k=k)
    print('Finished iterative stratification of development examples')

    print('Assigning examples to folds...')
    for l in labels_to_train + ['Background']:
        print(l)
        for f in range(k):
            dl = d[d['L'].apply(lambda x: l in x)]
            dl = dl[dl['fold'] == f]
    print(d)
    print(d['fold'].value_counts())

    print(f'Beginning {k}-fold dataset preparation')
    # Split each fold into validation data (e.g. 25 examples for labels with 125 total) and training data (e.g. 100 examples for labels with 125 total) with a split (e.g. 80/20), each fold being of up to a size (e.g. 25):
    # For example, if k=5:
    #  split 1: V T T T T
    #  split 2: T V T T T
    #  split 3: T T V T T
    #  split 4: T T T V T
    #  split 5: T T T T V

    # For each split...
    for s in range(k):
        print(f'Preparing datasets for split {s}...')

        # Get the validation fold for this split
        if k > 1:
            fold_idx_validation = s
        else:
            fold_idx_validation = -1 # i.e. no validation folds

        # Get the training folds for this split
        if training_set_proportion != 1.0:
            folds_idx_training  = [f for f in range(k) if f != fold_idx_validation]
            shared_validation_data = d[d['fold'] == fold_idx_validation]
            shared_validation_data = shared_validation_data.copy()
        else:
            folds_idx_training = [f for f in range(k)]
            shared_validation_data = pd.DataFrame()
        print(f'Training folds {folds_idx_training}')

        available_training_data = d[d['fold'].isin(folds_idx_training)]
        validation_class_counts = class_imbalance_test(shared_validation_data, print_out=True)

        # For each sample size...
        train_samples = pd.DataFrame()
        for sample_size in sample_sizes:

            print(f'sample_size {sample_size}')

            model_iteration_id_stub = f'{target_model_stub}_N{sample_size}_I{s}'
            path_model_out = f'models/target/{target_model_stub}/{model_iteration_id_stub}'
            file_model_out = f'{path_model_out}/{model_iteration_id_stub}.tflite'
            os.makedirs(path_model_out, exist_ok=True)

            # Randomly sample the required number of examples per label from the training folds for the split
            for l in (labels_to_train + ['Background']):
                print(f'l {l}')
                l_examples = available_training_data[available_training_data['L'].apply(lambda x: l in x)]
                print(f'{len(l_examples)} examples')
                training_sample = l_examples.sample(n=min(sample_size, len(l_examples)), random_state=training_seed)
                print(f'training_sample {training_sample}')
                available_training_data = available_training_data.drop(training_sample.index) # Remove the samples from 'available_training_data'
                train_samples = pd.concat([train_samples, training_sample]).reset_index(drop=True)

            sample_class_counts = class_imbalance_test(train_samples, print_out=True)

            # For labels with fewer than the required number of examples, optionally upsample to artificially increase the number of examples via repeat.
            if upsample:
                labels_to_upsample = sample_class_counts[sample_class_counts < sample_size].index.tolist()
                print(f'Upsampling {len(labels_to_upsample)} labels:')
                for label_to_upsample in labels_to_upsample:
                    label_examples = train_samples[train_samples['L'].apply(lambda x: label_to_upsample in x)]
                    n = sample_size - len(label_examples)
                    print(f'Augmenting {label_to_upsample} (N={len(label_examples)}) with {n} random repeats...')
                    upsampled_samples = label_examples.sample(n=n, replace=True, random_state=training_seed)
                    train_samples = pd.concat([train_samples, upsampled_samples], ignore_index=True) # add to development_examples

            # Store combined training and validation filepaths
            combined_files_csv_path = os.path.abspath(f'{output_path}/{model_iteration_id_stub}/combined_development_files.csv')
            os.makedirs(f'{output_path}/{model_iteration_id_stub}', exist_ok=True)
            train_samples['dataset'] = 'training'
            if training_set_proportion != 1.0:
                shared_validation_data['dataset'] = 'validation'
            combined_examples = pd.concat([train_samples, shared_validation_data], axis=0)
            combined_examples.to_csv(combined_files_csv_path, index=False)

            # Output command to train the model on these samples 
            print(f'Model {os.path.basename(file_model_out)} prepared for training with {sample_size} examples ======================================================================================')
            command = [
                'python3', 'train_fewshot.py',
                '--i', os.path.abspath(combined_files_csv_path), # Path to combined (training and validation) data references csv.
                '--l', os.path.abspath(target_species_list_file), # Path to class labels txt.
                '--o', os.path.abspath(file_model_out), # File path to trained classifier model output.
                '--no-autotune' if True else '--autotune', # Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).\
                '--epochs', str(epochs),
                '--learning_rate', str(learning_rate),
                '--batch_size',    str(batch_size),
                '--hidden_units',  str(hidden_units)
            ]

            print('Manually execute the following commands to begin training:\n')
            print('cd src/submodules/BirdNET-Analyzer')
            print(' '.join(command))
            print()
