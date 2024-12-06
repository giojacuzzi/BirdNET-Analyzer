"""
Module for training a custom classifier with prespecified training and validation data.
Adapted by Giordano Jacuzzi from BirdNET-Analyzer train.py
"""
import argparse
import multiprocessing
import os
from functools import partial
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np
import tqdm

import audio
import config as cfg
import model
import utils
import time
import sys


def _loadAudioFile(f, label_vector, config):
    """Load an audio file and extract features.
    Args:
        f: Path to the audio file.
        label_vector: The label vector for the file.
    Returns:
        A tuple of (x_train, y_train).
    """    

    x_train = []
    y_train = []

    # restore config in case we're on Windows to be thread save
    cfg.setConfig(config)

    # Try to load the audio file
    try:
        # Load audio
        sig, rate = audio.openAudioFile(f, duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX)

    # if anything happens print the error and ignore the file
    except Exception as e:
        # Print Error
        print(f"\t Error when loading file {f}", flush=True)
        return np.array([]), np.array([])

    # Crop training samples
    if cfg.SAMPLE_CROP_MODE == "center":
        sig_splits = [audio.cropCenter(sig, rate, cfg.SIG_LENGTH)]
    elif cfg.SAMPLE_CROP_MODE == "first":
        sig_splits = [audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
    else:
        sig_splits = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    # Get feature embeddings
    batch_size = 1 # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes
    for i in range(0, len(sig_splits), batch_size):
        batch_sig = sig_splits[i:i+batch_size]
        batch_label = [label_vector] * len(batch_sig)
        embeddings = model.embeddings(batch_sig)

        # Add to training data
        x_train.extend(embeddings)
        y_train.extend(batch_label)

    return x_train, y_train, [f] # Return audio file name

def _loadTrainingData(cache_mode="none", cache_file="", progress_callback=None):
    """Loads the data for training.

    Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names as new labels.

    These directories should contain all the training data for each label.

    If a cache file is provided, the training data is loaded from there.

    Args:
        cache_mode: Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.
        cache_file: Path to cache file.

    Returns:
        A tuple of (x_train, y_train, labels).
    """
    # Load from cache
    if cache_mode == "load":
        if os.path.isfile(cache_file):
            print(f"\t...loading from cache: {cache_file}", flush=True)
            x_train, y_train, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = utils.loadFromCache(cache_file)
            return x_train, y_train, labels
        else:
            print(f"\t...cache file not found: {cache_file}", flush=True)

    development_files = pd.read_csv(cfg.TRAIN_DATA_PATH)
    train_files = development_files[development_files['dataset'] == 'training']
    validation_files = development_files[development_files['dataset'] == 'validation']

    print('Valid labels:')
    with open(cfg.TRAIN_LABELS_PATH, 'r') as file:
        valid_labels = [label.strip() for label in file if label.strip()]
    print(valid_labels)

    print('All annotated labels:')
    all_labels = []
    label_combinations = development_files['labels'].unique()
    for combo in label_combinations:
        labels = combo.split(', ')
        for label in labels:
            if not label in all_labels:
                all_labels.append(str(label))
    all_labels = list(sorted(all_labels))
    print(all_labels)

    # Check if binary classification
    cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

    # Validate the classes for binary classification
    if cfg.BINARY_CLASSIFICATION:
        if len([l for l in all_labels if l.startswith("-")]) > 0:
            raise Exception("Negative labels cant be used with binary classification")
        if len([l for l in all_labels if l.lower() in cfg.NON_EVENT_CLASSES]) == 0:
            raise Exception("Non-event samples are required for binary classification")

    # Check if multi label
    cfg.MULTI_LABEL = len(valid_labels) > 1 and any(',' in f for f in label_combinations)

    # Check if multi-label and binary classficication 
    if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
        raise Exception("Error: Binary classfication and multi-label not possible at the same time")

    # Only allow repeat upsampling for multi-label setting
    if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != 'repeat':
        raise Exception("Only repeat-upsampling ist available for multi-label")

    # Load training and validation data
    x_train = []
    y_train = []

    x_val = []
    y_val = []

    n_train_files_loaded = 0
    n_val_files_loaded = 0
    for label_combo in label_combinations:
        print(f'Loading data for {label_combo}...')

        # Get label vector
        label_vector = np.zeros((len(valid_labels),), dtype="float32")

        combo_labels = label_combo.split(', ')

        for label in combo_labels:
            if (not label.lower() in cfg.NON_EVENT_CLASSES) and (not label.startswith("-")) and label in valid_labels:
                label_vector[valid_labels.index(label)] = 1
            elif label.startswith("-") and label[1:] in valid_labels: # Negative labels need to be contained in the valid labels
                label_vector[valid_labels.index(label[1:])] = -1
            # Note that labels that are not included in valid_labels will be ignored

        # Load training files using thread pool    
        train_files_to_load = train_files[train_files['labels'] == label_combo]['path']
        print(f'Loading {len(train_files_to_load)} training files using thread pool...')  
        with Pool(cfg.CPU_THREADS) as p:
            tasks = []
            for f in train_files_to_load:
                task = p.apply_async(partial(_loadAudioFile, f=f, label_vector=label_vector, config=cfg.getConfig()))
                tasks.append(task)
            num_files_processed = 0 # Wait for tasks to complete and monitor progress with tqdm
            with tqdm.tqdm(total=len(tasks), desc=f" - loading '{label_combo}'", unit='f') as progress_bar:
                for task in tasks:
                    result = task.get()
                    x_train += result[0]
                    y_train += result[1]
                    num_files_processed += 1
                    progress_bar.update(1)
                    if progress_callback:
                        progress_callback(num_files_processed, len(tasks), label_combo)
        n_train_files_loaded = n_train_files_loaded + len(train_files_to_load)
        print(f'{n_train_files_loaded}/{len(train_files)} ({round(n_train_files_loaded/len(train_files) * 100,2)}%) training files loaded')

        # Load validation files using thread pool
        val_files_to_load = validation_files[validation_files['labels'] == label_combo]['path']
        if len(validation_files) > 0:
            print(f'Loading {len(val_files_to_load)} validation files using thread pool...')
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []
                for f in val_files_to_load:
                    task = p.apply_async(partial(_loadAudioFile, f=f, label_vector=label_vector, config=cfg.getConfig()))
                    tasks.append(task)
                num_files_processed = 0 # Wait for tasks to complete and monitor progress with tqdm
                with tqdm.tqdm(total=len(tasks), desc=f" - loading '{label_combo}'", unit='f') as progress_bar:
                    for task in tasks:
                        result = task.get()
                        x_val += result[0]
                        y_val += result[1]
                        num_files_processed += 1
                        progress_bar.update(1)
                        if progress_callback:
                            progress_callback(num_files_processed, len(tasks), label_combo)
            n_val_files_loaded = n_val_files_loaded + len(val_files_to_load)
            pcnt_complete = n_val_files_loaded/len(validation_files) * 100
        else:
            print('No validation files to load')
            pcnt_complete = 100
        print(f'{n_val_files_loaded}/{len(validation_files)} ({round(pcnt_complete,2)}%) validation files loaded')
    
    # Convert to numpy arrays
    x_train = np.array(x_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")
    x_val = np.array(x_val, dtype="float32")
    y_val = np.array(y_val, dtype="float32")
    
    # Save to cache?
    # NOTE: Cache not supported!
    if cache_mode == "save":
        print(f"\t...saving training data to cache: {cache_file}", flush=True)
        try:
            # Only save the valid labels
            utils.saveToCache(cache_file, x_train, y_train, valid_labels)
        except Exception as e:
            print(f"\t...error saving cache: {e}", flush=True)

    # Return only the valid labels for further use
    return x_train, y_train, x_val, y_val, valid_labels


def trainModel(on_epoch_end=None, on_trial_result=None, on_data_load_end=None):
    """Trains a custom classifier.

    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.

    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """

    # Load training data
    print("Loading training data...", flush=True)
    x_train, y_train, x_val, y_val, labels = _loadTrainingData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
    print(f"...Done. Loaded {x_train.shape[0]} training samples and {x_val.shape[0]} validation samples across {len(labels)} labels.", flush=True)

    if cfg.AUTOTUNE:
        import gc

        import keras
        import keras_tuner

        # Call callback to initialize progress bar
        if on_trial_result:
            on_trial_result(0)

        class BirdNetTuner(keras_tuner.BayesianOptimization):
            def __init__(self, x_train, y_train, x_val, y_val, max_trials, executions_per_trial, on_trial_result):
                super().__init__(max_trials=max_trials, executions_per_trial=executions_per_trial, overwrite=True, directory="autotune", project_name="birdnet_analyzer")
                self.x_train = x_train
                self.y_train = y_train
                self.x_val = x_val
                self.y_val = y_val
                self.on_trial_result = on_trial_result

            def run_trial(self, trial, *args, **kwargs):
                histories = []
                hp: keras_tuner.HyperParameters = trial.hyperparameters
                trial_number = len(self.oracle.trials)

                for execution in range(int(self.executions_per_trial)):
                    print(f"Running Trial #{trial_number} execution #{execution + 1}", flush=True)

                    # Build model
                    print("Building model...", flush=True)
                    classifier = model.buildLinearClassifier(self.y_train.shape[1], 
                                                            self.x_train.shape[1], 
                                                            hidden_units=hp.Choice("hidden_units", [0, 128, 256, 512, 1024, 2048], default=cfg.TRAIN_HIDDEN_UNITS), 
                                                            dropout=hp.Choice("dropout", [0.0, 0.25, 0.33, 0.5, 0.75, 0.9], default=cfg.TRAIN_DROPOUT))
                    print("...Done.", flush=True)

                    # Only allow repeat upsampling in multi-label setting
                    upsampling_choices = ['repeat', 'mean', 'linear'] #SMOTE is too slow
                    if cfg.MULTI_LABEL:
                        upsampling_choices = ['repeat']

                    # Train model
                    print("Training model...", flush=True)
                    classifier, history = model.trainLinearClassifier(
                        classifier,
                        self.x_train,
                        self.y_train,
                        self.x_val,
                        self.y_val,
                        epochs=cfg.TRAIN_EPOCHS,
                        batch_size=hp.Choice("batch_size", [8, 16, 32, 64, 128], default=cfg.TRAIN_BATCH_SIZE),
                        learning_rate=hp.Choice("learning_rate", [0.1, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], default=cfg.TRAIN_LEARNING_RATE),
                        val_split=cfg.TRAIN_VAL_SPLIT, # NOTE: Unused parameter. Validation data is explicitly specified to allow for later performance evaluation.
                        upsampling_ratio=hp.Choice("upsampling_ratio",[0.0], default=cfg.UPSAMPLING_RATIO), # 0.0, 0.25, 0.33, 0.5, 0.75, 1.0
                        upsampling_mode=hp.Choice("upsampling_mode", upsampling_choices, default=cfg.UPSAMPLING_MODE), 
                        train_with_mixup=False, # hp.Boolean("mixup", default=cfg.TRAIN_WITH_MIXUP),
                        train_with_label_smoothing=hp.Boolean("label_smoothing", default=cfg.TRAIN_WITH_LABEL_SMOOTHING),
                    )

                    # Get the best validation loss
                    # Is it maybe better to return the negative val_auprc??
                    best_val_loss = history.history["val_loss"][np.argmin(history.history["val_loss"])]
                    histories.append(best_val_loss)

                    print(f"Finished Trial #{trial_number} execution #{execution + 1}. best validation loss: {best_val_loss}", flush=True)

                keras.backend.clear_session()
                del classifier
                del history
                gc.collect()

                # Call the on_trial_result callback
                if self.on_trial_result:
                    self.on_trial_result(trial_number)

                return histories

        tuner = BirdNetTuner(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, max_trials=cfg.AUTOTUNE_TRIALS, executions_per_trial=cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL, on_trial_result=on_trial_result)
        tuner.search()
        best_params = tuner.get_best_hyperparameters()[0]
        print("Best params: ")
        print("hidden_units: ", best_params["hidden_units"])
        print("dropout: ", best_params["dropout"])
        print("batch_size: ", best_params["batch_size"])
        print("learning_rate: ", best_params["learning_rate"])
        print("upsampling_mode: ", best_params["upsampling_mode"])
        print("upsampling_ratio: ", best_params["upsampling_ratio"])
        # print("mixup: ", best_params["mixup"])
        print("label_smoothing: ", best_params["label_smoothing"])
        cfg.TRAIN_HIDDEN_UNITS = best_params["hidden_units"]
        cfg.TRAIN_DROPOUT = best_params["dropout"]
        cfg.TRAIN_BATCH_SIZE = best_params["batch_size"]
        cfg.TRAIN_LEARNING_RATE = best_params["learning_rate"]
        cfg.UPSAMPLING_MODE = best_params["upsampling_mode"]
        cfg.UPSAMPLING_RATIO = best_params["upsampling_ratio"]
        # cfg.TRAIN_WITH_MIXUP = best_params["mixup"]
        cfg.TRAIN_WITH_LABEL_SMOOTHING = best_params["label_smoothing"]
        

    # Build model
    print("Building model...", flush=True)
    classifier = model.buildLinearClassifier(y_train.shape[1], x_train.shape[1], cfg.TRAIN_HIDDEN_UNITS, cfg.TRAIN_DROPOUT)
    print("...Done.", flush=True)
    
    # Train model
    print("Training model...", flush=True)
    classifier, history = model.trainLinearClassifier(
        classifier,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=cfg.TRAIN_EPOCHS,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        learning_rate=cfg.TRAIN_LEARNING_RATE,
        val_split=cfg.TRAIN_VAL_SPLIT, # NOTE: Unused parameter. Validation data is explicitly specified to allow for later performance evaluation.
        upsampling_ratio=cfg.UPSAMPLING_RATIO,
        upsampling_mode=cfg.UPSAMPLING_MODE,
        train_with_mixup=cfg.TRAIN_WITH_MIXUP,
        train_with_label_smoothing=cfg.TRAIN_WITH_LABEL_SMOOTHING,
        on_epoch_end=on_epoch_end,
    )

    if cfg.TRAINED_MODEL_OUTPUT_FORMAT == "both":
        model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
        model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
    elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
        model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
    elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
        model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
    else:
        raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")
    print('Saved model outputs.')

    # Best validation AUPRC (at minimum validation loss)
    if x_val.shape[0] > 0:
        best_val_auprc = history.history["val_AUPRC"][np.argmin(history.history["val_loss"])]
        best_val_auroc = history.history["val_AUROC"][np.argmin(history.history["val_loss"])]
        print(f"...Done. Best AUPRC: {best_val_auprc}, Best AUROC: {best_val_auroc}", flush=True)

    return history


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a custom classifier with BirdNET")
    parser.add_argument("--i", default="train_data/", help="Path to training data references.")
    parser.add_argument("--l", default="train_data/", help="Path to training data labels csv.")
    parser.add_argument("--crop_mode", default="center", help="Crop mode for training data. Can be 'center', 'first' or 'segments'. Defaults to 'center'.")
    parser.add_argument("--crop_overlap", type=float, default=0.0, help="Overlap of training data segments in seconds if crop_mode is 'segments'. Defaults to 0.")
    parser.add_argument(
        "--o", default="checkpoints/custom/Custom_Classifier", help="Path to trained classifier model output."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs. Defaults to 50.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Defaults to 32.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio. Defaults to 0.2.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. Defaults to 0.001.")
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=0,
        help="Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Defaults to 0.")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, help="Whether to use mixup for training.")
    parser.add_argument("--upsampling_ratio", type=float, default=0.0, help="Balance train data and upsample minority classes. Values between 0 and 1. Defaults to 0.")
    parser.add_argument("--upsampling_mode", default="repeat", help="Upsampling mode. Can be 'repeat', 'mean' or 'smote'. Defaults to 'repeat'.")
    parser.add_argument("--model_format", default="tflite", help="Model output format. Can be 'tflite', 'raven' or 'both'. Defaults to 'tflite'.")
    parser.add_argument("--model_save_mode", default="replace", help="Model save mode. Can be 'replace' or 'append', where 'replace' will overwrite the original classification layer and 'append' will combine the original classification layer with the new one. Defaults to 'replace'.")
    parser.add_argument("--cache_mode", default="none", help="Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.")
    parser.add_argument("--cache_file", default="train_cache.npz", help="Path to cache file. Defaults to 'train_cache.npz'.")
    parser.add_argument("--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads.")

    parser.add_argument("--fmin", type=int, default=cfg.SIG_FMIN, help="Minimum frequency for bandpass filter in Hz. Defaults to {} Hz.".format(cfg.SIG_FMIN))
    parser.add_argument("--fmax", type=int, default=cfg.SIG_FMAX, help="Maximum frequency for bandpass filter in Hz. Defaults to {} Hz.".format(cfg.SIG_FMAX))

    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, help="Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).")
    parser.add_argument("--autotune_trials", type=int, default=50, help="Number of training runs for hyperparameter tuning. Defaults to 50.")
    parser.add_argument("--autotune_executions_per_trial", type=int, default=1, help="The number of times a training run with a set of hyperparameters is repeated during hyperparameter tuning (this reduces the variance). Defaults to 1.")

    args = parser.parse_args()

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.TRAIN_LABELS_PATH = args.l
    cfg.SAMPLE_CROP_MODE = args.crop_mode
    cfg.SIG_OVERLAP = args.crop_overlap
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_VAL_SPLIT = args.val_split
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units
    cfg.TRAIN_DROPOUT = min(max(0, args.dropout), 0.9)
    cfg.TRAIN_WITH_MIXUP = args.mixup if args.mixup is not None else cfg.TRAIN_WITH_MIXUP
    cfg.UPSAMPLING_RATIO = min(max(0, args.upsampling_ratio), 1)
    cfg.UPSAMPLING_MODE = args.upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = args.model_format
    cfg.TRAINED_MODEL_SAVE_MODE = args.model_save_mode
    cfg.TRAIN_CACHE_MODE = args.cache_mode.lower()
    cfg.TRAIN_CACHE_FILE = args.cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = max(1, int(args.threads))

    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(args.fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(args.fmax)))

    cfg.AUTOTUNE = args.autotune
    cfg.AUTOTUNE_TRIALS = args.autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = args.autotune_executions_per_trial

    # Train model
    history = trainModel()

    try:
        auprc = history.history["val_AUPRC"]
        auroc = history.history["val_AUROC"]
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(auprc, label="AUPRC")
        plt.plot(auroc, label="AUROC")
        plt.legend()
        plt.xlabel("Epoch")
        plt.show()
    except Exception as e:
        print(f"Unable to report validation results")
    finally:
        print("Finished training!")