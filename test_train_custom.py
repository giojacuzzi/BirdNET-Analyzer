# Assemble datasets for model training and performance evaluation
# Before running this script, run `training_extract_audio_examples.py` to extract audio examples to 'data/training/Custom'
# TODO: REPLACE THIS WITH train.py in the root directory of the src repo
if __name__ == "__main__":

    training_data_path = 'data/training'
    output_path = 'data/models/custom'

    sample_size_experiments = [100] # Sample size experiments for model development (e.g. 2, 5, 10, 25, 50, 75, 100)
    autotune = 0 # TODO: run another loop with autotune

    data_augmentation = 0 # TODO: Support augmentation

    upsample = 0

    test_set_size = 25 # for novel labels (25)
    development_set_size = 125 # training + validation total (125)
    # NOTE: Labels to train are specified in training_labels.csv

    stratified_kfold_cross_validation = True

    import random
    import pandas as pd
    import numpy as np
    import os
    import sys
    import subprocess
    import copy

    # Load class labels
    class_labels_csv_path = os.path.abspath(f'{training_data_path}/training_labels.csv')
    class_labels = pd.read_csv(class_labels_csv_path)
    class_labels = class_labels[class_labels['train'] == 1]
    preexisting_labels_to_train = list(class_labels[class_labels['novel'] == 0]['label_birdnet'])
    novel_labels_to_train = list(class_labels[class_labels['novel'] == 1]['label_birdnet'])
    labels_to_train = preexisting_labels_to_train + novel_labels_to_train

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
    training_data_audio_path = training_data_path + '/custom/audio'
    print(f'Finding all audio data for model development from {training_data_audio_path}...')
    training_filepaths = []
    training_filepaths.extend(find_files(training_data_audio_path, '.wav'))
    # print(list({os.path.dirname(path) for path in training_filepaths}))

    available_examples = pd.DataFrame()
    for path in training_filepaths:
        dir = os.path.basename(os.path.dirname(path))
        example = pd.DataFrame({
            'source': [os.path.basename(os.path.dirname(path))],
            'file':   [os.path.basename(path)],
            'path':   [os.path.abspath(path)]
        })
        available_examples = pd.concat([available_examples, example], ignore_index=True)

    # Load annotation labels for the training files and merge, retaining only those that match the available examples
    # TODO: Incoporate data augmentation option to either include or exclude the augmented examples as available_examples
    annotations = pd.read_csv(f'{training_data_path}/training_data_annotations.csv')
    annotations['file'] = annotations['file'] + '.wav'
    available_examples = available_examples.merge(annotations[['source', 'file', 'labels']], on=['source', 'file'], how='left')
    # print(available_examples['source'].value_counts())
    available_examples.loc[available_examples['source'] == 'Background', 'labels'] = 'Background' # manually label background examples

    # DO NOT CHANGE
    # Never change this random seed to ensure unbiased evaluation of test data.
    test_seed = 1

    # Randomly set aside 25 examples of each novel label as test data
    print(f'Randomly choosing {test_set_size} examples for each novel label (N={len(novel_labels_to_train)}) as test data...')
    test_examples_novel = pd.DataFrame()
    for novel_label in novel_labels_to_train:
        if novel_label == 'Background': # skip the Background label
            continue
        label_examples = available_examples[available_examples['source'] == novel_label]
        sampled_rows = label_examples.sample(n=test_set_size, random_state=test_seed)
        test_examples_novel = pd.concat([test_examples_novel, sampled_rows], ignore_index=True)
        available_examples = available_examples.drop(sampled_rows.index) # Remove the samples from the training data
    # print(test_examples_novel['label'].value_counts())

    # # Store test example filepaths
    # test_files_csv_path = os.path.abspath(f'{output_path}/test_files.csv')
    # test_examples_novel.to_csv(test_files_csv_path, index=False)

    # Find the majority and minority classes (i.e. value_couts across all labels present in available_examples)
    def class_imbalance_test(df_in, print_out=False):
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

    # Set a random seed
    training_seed = 1
    random.seed(training_seed)

    # Create a development (train + validation) dataset by randomly choosing 125 examples (or, if less than 125 exist, as many as are available) from the total available for each label.
    print(f'Randomly choosing {development_set_size} examples for each label as development data...')
    development_examples = pd.DataFrame()
    for label_to_train in (labels_to_train + ['Background']):
        # NOTE: this uses any label examples; to use targeted examples: available_examples[available_examples['source'] == label_to_train]
        label_examples = available_examples[available_examples['labels'].str.contains(label_to_train, regex=False)]
        if len(label_examples) < development_set_size:
            print(f'WARNING: Less than {development_set_size} examples available for label {label_to_train}')
        sampled_rows = label_examples.sample(n=min(development_set_size, len(label_examples)), random_state=training_seed)
        development_examples = pd.concat([development_examples, sampled_rows], ignore_index=True)
        available_examples = available_examples.drop(sampled_rows.index) # Remove the sampled examples from 'available_examples'
    print(f'Found {len(development_examples)} total development examples')

    print('DEVELOPMENT EXAMPLES:')
    class_counts = class_imbalance_test(development_examples, print_out=True)

    # ========================================================================================================================================================================================================
    # PREPARE TRAIN WITH NO CROSS VALIDATION
    if not stratified_kfold_cross_validation:
        print(f'Preparing data for training (no cross-validation) ======================================================================================')

        # Split development data into 20% validation data (25 examples for labels with 125 total) and 80% training data (100 examples for labels with 125 total).
        # This validation data will be used across all model experiments
        training_set_proportion = 0.80
        validation_set_proportion = round(1.0 - training_set_proportion, 2)
        print(f'Randomly choosing {training_set_proportion * 100}/{validation_set_proportion * 100}% for each label as training/validation data...')
        validation_examples = pd.DataFrame()
        training_examples   = pd.DataFrame()
        for label_to_train in (labels_to_train + ['Background']):
            # print(f'Development examples {len(development_examples)}')
            label_examples   = development_examples[development_examples['labels'].str.contains(label_to_train, regex=False)]
            label_validation = label_examples.sample(frac=validation_set_proportion, random_state=training_seed)
            label_training   = label_examples.drop(label_validation.index)
            development_examples = development_examples.drop(label_validation.index) # Remove the samples from 'development_examples'
            validation_examples = pd.concat([validation_examples, label_validation]).reset_index(drop=True)
            training_examples   = pd.concat([training_examples, label_training]).reset_index(drop=True)
        # print(validation_examples['source'].value_counts())
        # print(training_examples['source'].value_counts())

        print('TRAINING EXAMPLES:')
        class_counts = class_imbalance_test(training_examples, print_out=True)
        print('VALIDATION EXAMPLES:')
        class_counts = class_imbalance_test(validation_examples, print_out=True)

        # # Store validation example filepaths
        # validation_files_csv_path = os.path.abspath(f'{output_path}/validation_files.csv')
        # validation_examples.to_csv(validation_files_csv_path, index=False)

        # For each experiment (2,5,10,25,50,100 training examples)...
        for experiment_sample_size in sample_size_experiments:
            print(f'Running model development with {experiment_sample_size} training samples...')

            model_iteration_id_stub = f'custom_S{training_seed}_N{experiment_sample_size}_A{autotune}'
            path_model_out = f'{os.getcwd()}/{output_path}/{model_iteration_id_stub}'
            file_model_out = f'{path_model_out}/{model_iteration_id_stub}.tflite'
            os.makedirs(path_model_out, exist_ok=True)

            # Randomly sample the required number of examples per label from the training data
            experiment_examples = pd.DataFrame()
            for label_to_train in (labels_to_train + ['Background']):
                print(f'Training examples {len(training_examples)}')
                label_examples   = training_examples[training_examples['labels'].str.contains(label_to_train, regex=False)]
                training_sample  = label_examples.sample(n=min(experiment_sample_size, len(label_examples)), random_state=training_seed)
                training_examples = training_examples.drop(training_sample.index) # Remove the samples from 'training_examples'
                experiment_examples = pd.concat([experiment_examples, training_sample]).reset_index(drop=True)
            print('EXPERIMENT EXAMPLES BEFORE UPSAMPLING:')
            class_counts = class_imbalance_test(experiment_examples, print_out=True)

            # For labels with fewer than the required number of examples, upsample to artificially increase the number of examples via repeat.
            if upsample:
                labels_to_upsample = class_counts[class_counts < experiment_sample_size].index.tolist()
                print(labels_to_upsample)
                for label_to_upsample in labels_to_upsample:
                    label_examples = experiment_examples[experiment_examples['labels'].str.contains(label_to_upsample, regex=False)]
                    n = experiment_sample_size - len(label_examples)
                    print(f'Augmenting {label_to_upsample} (N={len(label_examples)}) with {n} random repeats...')
                    upsampled_samples = label_examples.sample(n=n, replace=True, random_state=training_seed)
                    experiment_examples = pd.concat([experiment_examples, upsampled_samples], ignore_index=True) # add to development_examples

            print('EXPERIMENT EXAMPLES AFTER UPSAMPLING:')
            class_counts = class_imbalance_test(experiment_examples, print_out=True)

            experiment_examples = experiment_examples.sort_values(by=['path'])
            # print(experiment_examples.to_string())

            # # Store training filepaths
            # training_files_csv_path = os.path.abspath(f'{output_path}/{model_iteration_id_stub}/training_files.csv')
            # experiment_examples.to_csv(training_files_csv_path, index=False)

            # Store combined training and validation filepaths
            combined_files_csv_path = os.path.abspath(f'{output_path}/{model_iteration_id_stub}/combined_files.csv')
            experiment_examples['dataset'] = 'training'
            validation_examples['dataset'] = 'validation'
            combined_examples = pd.concat([experiment_examples, validation_examples], axis=0)
            combined_examples.to_csv(combined_files_csv_path, index=False)

            # Change current working directory
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            print(f'cd {os.getcwd()}')

            # Train the model on these samples 
            print(f'Model {os.path.basename(file_model_out)} prepared for training with {experiment_sample_size} examples ======================================================================================')
            command = [
                'python3', 'train_custom.py',
                '--i', combined_files_csv_path, # Path to combined (training and validation) data references csv.
                '--l', class_labels_csv_path, # Path to class labels csv.
                '--o', file_model_out, # File path to trained classifier model output.
                '--no-autotune' if not autotune else '--autotune' # Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).
            ]

            print('Manually execute the following commands to begin training:')
            print('cd src/submodules/BirdNET-Analyzer')
            print(" ".join(command))
            print()
            print(f'Afterwards, execute test_compare_validation_performance.py with model {model_iteration_id_stub} to evaluate performance.')

            # Run the script with arguments
            # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # try: # Read output line by line in real-time
            #     for line in process.stdout:
            #         print(line, end='')  # Print the output as it is produced
            #     for line in process.stderr:
            #         print(line, end='', file=sys.stderr)  # Print errors as they occur
            # finally:
            #     process.wait()
            # print("Training finished with exit code:", process.returncode)
            
            # Evaluate model performance with the shared validation data
            # TODO

    # ========================================================================================================================================================================================================
    # PREPARE TRAIN WITH STRATIFIED K-FOLD CROSS VALIDATION
    else:
        print(f'Preparing data for training (stratified k-fold cross-validation) ======================================================================================')

        labels = (labels_to_train + ['Background'])

        def evenly_distribute_n_into_parts(n, parts):
            quotient, remainder = divmod(n, parts)
            result = [quotient] * parts
            for i in range(remainder): result[i] += 1
            return result

        # Iterative stratification http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
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

            # The algorithm will finish as soon as the original dataset gets empty
            d['fold'] = -1 # initialize all examples to be unassigned to a fold
            while (-1 in d['fold'].values):
                # Find the label with the fewest (but at least one) remaining examples, breaking ties randomly. It examines one label in each iteration, the one with the fewest remaining examples, denoted l.
                label_counts = class_imbalance_test(d[d['fold'] == -1])
                print(label_counts)
                # input()
                min_count_labels = label_counts[label_counts == label_counts.min()].index.tolist()
                # print(min_count_labels)
                l = random.choice(min_count_labels)
                print(f'Distributing: {l}')

                # Then, for each example (x, Y ) of this label dl, the algorithm selects an appropriate subset (fold) for distribution. The first criterion for subset (fold) selection is the current desired number of examples for this label clj. The subset (fold) that maximizes it gets selected.
                dl = d[d['L'].apply(lambda x: l in x)] # get all examples of the label
                dl = dl[dl['fold'] == -1] # get only examples that have yet to be assigned to a fold
                # print(f'dl {dl.to_string()}')
                for index, example in dl.iterrows():
                    # print(f'{example}')

                    # Select a fold for distribution of the sample, prioritizing the fold with the maximum number of remaining desired examples of this label
                    clj = cl[cl['label'] == l]['clj'].iloc[0]
                    max_clj_idx = [i for i, v in enumerate(clj) if v == max(clj)]
                    
                    # TODO: In case of ties, then among the tying subsets (folds), the one with the highest number of desired examples cj get selected.  Further ties are broken randomly 
                    if len(max_clj_idx) > 1:
                        # print(f"Tie detected for label '{l}' at indices {max_clj_idx}.")
                        # Among the tying subsets (folds), the one with the highest number of desired examples cj get selected
                        # print(c)
                        max_c_idx = [i for i, v in enumerate(c) if v == max(c)]
                        if len(max_c_idx) > 1:
                            # print(f"Tie detected for folds {max_c_idx}.")
                            m = random.choice(max_clj_idx)
                            # print(f"Randomly selected index: {m}")
                        else:
                            m = max_c_idx[0]
                            # print(f"Selected fold priority index: {m}")

                    else:
                        m = max_clj_idx[0]
                        # print(f"Selected label priority index: {m}")
                    
                    # If the selected fold no longer desires more examples overall (although it may desire more of this specific label),
                    # choose the fold that desires the most examples overall instead
                    if c[m] == 0:
                        max_c_idx = [i for i, v in enumerate(c) if v == max(c)]
                        if len(max_c_idx) > 1:
                            m = random.choice(max_c_idx)
                        else:
                            m = max_c_idx[0]

                    # Once the appropriate subset, m, is selected, we add the example (x, Y ) to Sm and remove it from D.
                    d.loc[index, 'fold'] = m

                    # TODO: In the end of the iteration, we decrement the number of desired examples for each label of this example at subset m, cim, as well as the total number of desired examples for subset m, cm
                    cl[cl['label'] == l]['clj'].iloc[0][m] -= 1
                    c[m] -= 1

                print(d[d['L'].apply(lambda x: l in x)]['fold'].value_counts())
                print(f'Remaining desired number of total samples for each fold: {c}')

            return(d)

        print('BEGINNING ITERATIVE STRATIFICATION')
        k = 5
        d = iterative_stratification(d=development_examples, k=k)
        print('FINISHED ITERATIVE STRATIFICATION')
        for l in labels_to_train + ['Background']:
            print(l)
            for f in range(k):
                dl = d[d['L'].apply(lambda x: l in x)]
                dl = dl[dl['fold'] == f]
                print(f'Fold {f} - {len(dl)}')
        print(d)
        print(d['fold'].value_counts())

        print('BEGINNING K-FOLD CROSS-VALIDATION DATASET PREPARATION')
        # Split each fold into validation data (25 examples for labels with 125 total) and training data (100 examples for labels with 125 total) with a 80/20 split, each fold being of up to size 25:
        # 	split 1: V T T T T
        # 	split 2: T V T T T
        # 	split 3: T T V T T
        # 	split 4: T T T V T
        # 	split 5: T T T T V

        # For each split...
        for s in range(k):
            print(f'Preparing datasets for split {s}...')

            # Get the validation fold for this split
            fold_idx_validation = s
            # print(f'validation fold {fold_validation}')
            # Get the training folds for this split
            folds_idx_training  = [f for f in range(k) if f != fold_idx_validation]
            # print(f'training folds {folds_training}')

            available_training_data = d[d['fold'].isin(folds_idx_training)]

            shared_validation_data = d[d['fold'] == fold_idx_validation]
            shared_validation_data = shared_validation_data.copy()
            validation_class_counts = class_imbalance_test(shared_validation_data, print_out=False)

            # For each experiment sample size (2,5,10,25,50,100)...
            experiment_train_samples = pd.DataFrame()
            for experiment_sample_size in sample_size_experiments:

                model_iteration_id_stub = f'custom_S{training_seed}_N{experiment_sample_size}_A{autotune}_U{upsample}_I{fold_idx_validation}'
                path_model_out = f'{os.getcwd()}/{output_path}/{model_iteration_id_stub}'
                file_model_out = f'{path_model_out}/{model_iteration_id_stub}.tflite'
                os.makedirs(path_model_out, exist_ok=True)

                # Randomly sample the required number of examples per label from the training folds for the split
                for l in (labels_to_train + ['Background']):
                    l_examples = available_training_data[available_training_data['L'].apply(lambda x: l in x)]
                    # TODO: For labels with fewer than the required number of training examples, apply upsampling to artificially increase the number of examples

                    training_sample = l_examples.sample(n=min(experiment_sample_size, len(l_examples)), random_state=training_seed)
                    available_training_data = available_training_data.drop(training_sample.index) # Remove the samples from 'available_training_data'
                    # print(f'available training data: {len(available_training_data)}')
                    experiment_train_samples = pd.concat([experiment_train_samples, training_sample]).reset_index(drop=True)

                # print(f'SPLIT {s} TRAIN ----------------------------------------')
                experiment_class_counts = class_imbalance_test(experiment_train_samples, print_out=False)

                # Store combined training and validation filepaths
                combined_files_csv_path = os.path.abspath(f'{output_path}/{model_iteration_id_stub}/combined_files.csv')
                experiment_train_samples['dataset'] = 'training'
                # print(experiment_train_samples)
                shared_validation_data['dataset'] = 'validation'
                # print(shared_validation_data)
                combined_examples = pd.concat([experiment_train_samples, shared_validation_data], axis=0)
                combined_examples.to_csv(combined_files_csv_path, index=False)

                # Train the model on these samples 
                print(f'Model {os.path.basename(file_model_out)} prepared for training with {experiment_sample_size} examples ======================================================================================')
                command = [
                    'python3', 'train_custom.py',
                    '--i', combined_files_csv_path, # Path to combined (training and validation) data references csv.
                    '--l', class_labels_csv_path, # Path to class labels csv.
                    '--o', file_model_out, # File path to trained classifier model output.
                    '--no-autotune' if not autotune else '--autotune' # Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).
                ]

                print('Manually execute the following commands to begin training:')
                print('cd src/submodules/BirdNET-Analyzer')
                print(" ".join(command))
                print()
                print(f'Afterwards, execute test_compare_validation_performance.py with model {model_iteration_id_stub} to evaluate performance, then average across all iterations')
