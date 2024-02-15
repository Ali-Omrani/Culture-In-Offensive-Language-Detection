import logging
import numpy as np
import random
import torch
import datetime
import os
import pandas as pd
import IPython
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import TrainerCallback
from copy import deepcopy
from custom_trainer import CustomTrainer, CustomTrainerWithConstantNoiseMatrix, CustomTrainerFocalLoss

DATA_DIR = "../Data"

dataset2label = {
    "personal": ["tpa", "oa", "ra"],
    "ucc": ["antagonize", "condescending", "hostile"],
    "ghc":     ["vo", "hd", "cv"],
    "imdb":     ["sentiment"],
    "agnews":     ["category"],
    "arabic":     ["abuse"],
    "chinese":     ["offensive"],
    "russian":     ["toxicity"],
    "english":     ["hate"],
    "albanian":     ["offensive"],
    "danish":     ["offensive"],
    "estonian": ["moderated"],
    "german": ["offensive"],
    "greek": ["offensive"],
    "italian": ["hate"],
    "latvian": ["moderated"],
    "portuguese": ["hate"],
    "turkish": ["offensive"],
    "ukraninan": ["abusive"],
    "hindi": ["hate"],

}

label2dataset = { label:dataset for dataset, labels in dataset2label.items() for label in labels}






def get_trainer(trainer_type=None):
    if trainer_type == "constant_noise_matrix":
        return CustomTrainerWithConstantNoiseMatrix
    elif trainer_type == "withfocalloss":
        return CustomTrainerFocalLoss
    else:
        return CustomTrainer


def add_label_noise(example, noise_chance, label_col="label"):
    # Add a column to record whether the label was flipped or not
    example['label_flipped'] = np.random.rand() < noise_chance

    # Flip the label based on the noise_chance
    example[label_col] = 1 - example[label_col] if example['label_flipped'] else example[label_col]

    return example


def introduce_noise(df, label_col, noise_ratio):
    """
    Introduce noise to the label column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - label_col (str): Name of the label column to introduce noise.
    - noise_ratio (float): Ratio of rows to introduce noise to (value between 0 and 1).

    Returns:
    - pd.DataFrame: DataFrame with introduced noise in the label column.
    """

    # Validate the input parameters
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in the DataFrame.")

    if not (0 <= noise_ratio <= 1):
        raise ValueError("Noise ratio must be a value between 0 and 1.")

    # Determine the number of rows to introduce noise to
    num_rows = int(noise_ratio * len(df))

    # Randomly select rows to introduce noise
    noisy_rows = np.random.choice(df.index, size=num_rows, replace=False)

    # Flip the labels in the selected rows
    df.loc[noisy_rows, label_col] = 1 - df.loc[noisy_rows, label_col]

    return df


class EvalOnTrainCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="")
            return control_copy



class LogPredicitonsCallback(TrainerCallback):
    def __init__(self, logger, trainer, output_dir) -> None:
        super().__init__()
        self._logger = logger
        self._trainer = trainer
        self.output_dir = output_dir
    def on_epoch_end(self, args, state, control, **kwargs):
        print("in here logging predictions")
       
        for split in ["train", "val"]:
            if split == "train":
                dataset = self._trainer.train_dataset
            else:
                dataset = self._trainer.eval_dataset

            predictions = self._trainer.predict(test_dataset=dataset).predictions


            preds = np.argmax(predictions, axis=1)
            if os.path.exists(os.path.join(self.output_dir, f"{split}_preds.csv")):
                self._logger.info(f"Loading prev preds!!")
                predictions_df = pd.read_csv(os.path.join(self.output_dir, f"{split}_preds.csv"))
                pred_columns_ids = [int(c.split("_")[-1]) for c in predictions_df.columns if "pred" in c]
                pred_col = f"pred_{max(pred_columns_ids)+1}"
                self._logger.info(f"Adding predictions to column {pred_col} to {split}_preds.csv") 
                predictions_df[pred_col] = preds
                self._logger.info(f"{split}: f1 score of prev preds = {f1_score(predictions_df['label'], preds)}")
            else:
                self._logger.info(f"First epoch, no prev preds")
                labels = dataset['label']
                try:
                    flipped_labels = dataset['label_flipped']
                    ids = dataset['__index_level_0__']
                except:
                    self._logger.info(f"No label_flipped column for split {split}")
                    self._logger.info(f"No index_level_0 ids column for split {split}")
                    flipped_labels = [False]*len(labels)
                    ids = [i for i in range(len(labels))]
                
                pred_col = "pred_1"
                predictions_df = pd.DataFrame({"id":ids, "label_flipped":flipped_labels, "label":labels,  pred_col:preds})
            predictions_df.to_csv(os.path.join(self.output_dir, f"{split}_preds.csv"), index=False)

def compute_metrics(p):
    # Convert probabilities to predictions
    predictions = np.argmax(p.predictions, axis=1)
    
    # Assuming the labels are 0 and 1
    labels = p.label_ids

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # Use probabilities of class 1 for AUC-ROC
    auc_roc = roc_auc_score(labels, p.predictions[:, 1])
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
    }

def load_italian_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Italian", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Italian", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Italian", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_german_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "German", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "German", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "German", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_estonian_data():
    def fix_df(df):
        df.rename(columns={"content": "text"}, inplace=True)
        return df.dropna()
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Estonian", "train.csv")).reset_index()
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Estonian", "test.csv")).reset_index()
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Estonian", "val.csv")).reset_index()
    return {"train": fix_df(train_data), "test": fix_df(test_data), "val": fix_df(valid_data)}

def load_danish_data():
    def fix_df(df):
        df["offensive"] = train_data["label"].apply(lambda x: 1 if x == "OFF" else 0)
        return df[["offensive", "text"]]
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Danish", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Danish", "Danish_1_test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Danish", "val.csv"))
    return {"train": fix_df(train_data), "test": fix_df(test_data), "val": fix_df(valid_data)}


def load_albanian_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Albanian", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Albanian", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Albanian", "dev.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_stromfront_data():
    def fix_df(df):
        df["label"] = df["label"].apply(lambda x: 1 if x == "hate" else 0)
        df = df[["text", "label"]].rename(columns={"label": "hate"})
        return df
    data_path = os.path.join(DATA_DIR, "stormfront")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    return {"train": fix_df(train_df),  "val":fix_df(val_df), "test": fix_df(test_df)}

def load_chinese_data():
    def fix_df(df):
        return df[["TEXT", "label"]].rename(columns={"TEXT": "text", "label": "offensive"}).dropna()
    data_path = os.path.join(DATA_DIR, "Chinese")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    
    return {"train": fix_df(train_df),  "val":fix_df(val_df), "test": fix_df(test_df)}

def load_russian_data():
    def fix_df(df):
        df=df[["comments", "toxicity"]].rename(columns={"comments": "text"})
        return df.dropna()

    train = pd.read_csv("../Data/Russian/train.csv")
    test = pd.read_csv("../Data/Russian/test.csv")
    val = pd.read_csv("../Data/Russian/valid.csv")
    return {"train": fix_df(train), "test": fix_df(test), "val": fix_df(val)}

def load_arabic_data():
    def fix_df(df):
    
        # df.rename(columns={"Tweet":"text", "Class":"abuse"}, inplace=True)
        # df["abuse"] = train_df["abuse"].apply(lambda x: 1 if x=="abusive" else 0)
        return df.dropna()
    data_path = os.path.join(DATA_DIR, "Arabic")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    
    return {"train": fix_df(train_df), "test": fix_df(test_df), "val":fix_df(val_df)}


def load_ghc_data():
    data_path = "../../../Data/GHC/"

    train_df = pd.read_csv(os.path.join(data_path , "train.csv")).dropna()
    val_df = pd.read_csv(os.path.join(data_path , "valid.csv")).dropna()
    test_df = pd.read_csv(os.path.join(data_path , "test.csv")).dropna()
    return {"train":train_df, "val":val_df, "test":test_df}
    # return pd.concat([train_df, val_df, test_df])
 
def load_personal_attack_data():
    data_path = "../../../Data/personal_attack/"

    train_df = pd.read_csv(os.path.join(data_path , "train.csv")).rename(columns={"comment":"text"}).dropna()
    val_df = pd.read_csv(os.path.join(data_path , "dev.csv")).rename(columns={"comment":"text"}).dropna()
    test_df = pd.read_csv(os.path.join(data_path , "test.csv")).rename(columns={"comment":"text"}).dropna()
    # return pd.concat([train_df, val_df, test_df])
    return {"train":train_df, "val":val_df, "test":test_df}


def load_ucc_data():
    data_path = "../../../Data/ucc/"

    train_df = pd.read_csv(os.path.join(data_path , "train.csv")).rename(columns={"comment":"text"}).dropna()
    val_df = pd.read_csv(os.path.join(data_path , "dev.csv")).rename(columns={"comment":"text"}).dropna()
    test_df = pd.read_csv(os.path.join(data_path , "test.csv")).rename(columns={"comment":"text"}).dropna()
    return {"train":train_df, "val":val_df, "test":test_df}
    # return pd.concat([train_df, val_df, test_df])


def load_jigsaw_data():
    data_dir = "../../Data/Jigsaw Annotations/8670-items(re-annotated one VO questions version)"
    dfs = []
    for i in range(1, 6):
        annotator_file = f"annotator{i}.json"
        annotator_path = os.path.join(data_dir, annotator_file)
        annotator_df = pd.read_json(annotator_path, lines=True)
        annotator_df['annotator'] = i
        dfs.append(annotator_df)

    all_annotations_df = pd.concat(dfs)
    all_annotations_df
    LABELS = ['CV', 'HD', 'VO', 'NH', 'RAE', 'NAT', 'GEN', 'REL', 'SXO', 'IDL', 'POL', 'MPH', 'Explicit', 'Implicit']
    all_annotations_df['accept'] = all_annotations_df['accept'].apply(lambda x: x if isinstance(x, list) else [])
    # Create new columns for each value in LABELS
    for label in LABELS:
        all_annotations_df[label] = all_annotations_df['accept'].apply(lambda x: 1 if label in x else 0)

    return all_annotations_df

def load_imdb_data():
    data_path = "../Data/IMDB"

    train_df = pd.read_csv(os.path.join(data_path , "train.csv")).rename(columns={'review': 'text'})
    val_df = pd.read_csv(os.path.join(data_path , "valid.csv")).rename(columns={'review': 'text'})
    test_df = pd.read_csv(os.path.join(data_path , "test.csv")).rename(columns={'review': 'text'})

    sentiment_mapping = {'positive': 1, 'negative': 0}
    for df in [train_df, test_df, val_df]:
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    return {"train":train_df, "val":val_df, "test":test_df}


def load_agnews_data():
    data_path = "../../../Data/AGNEWS"

    train_df = pd.read_csv(os.path.join(data_path , "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path , "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path , "test.csv"))
    return {"train":train_df, "val":val_df, "test":test_df}

def load_greek_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Greek", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Greek", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Greek", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_latvian_data():
    def fix_df(df):
        df.rename(columns={"content": "text"}, inplace=True)
        return df
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Latvian", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Latvian", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Latvian", "val.csv"))
    return {"train": fix_df(train_data), "test": fix_df(test_data), "val": fix_df(valid_data)}

def load_portuguese_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Portuguese", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Portuguese", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Portuguese", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_turkish_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Turkish", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Turkish", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Turkish", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_ukranian_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Ukranian", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Ukranian", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Ukranian", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def load_hindi_data():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "Hindi", "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "Hindi", "test.csv"))
    valid_data = pd.read_csv(os.path.join(DATA_DIR, "Hindi", "val.csv"))
    return {"train": train_data, "test": test_data, "val": valid_data}

def get_dataset_loader_func(dataset_name):
    
    if dataset_name == "personal_attack":
        return load_personal_attack_data()
    elif dataset_name == "imdb":
        return load_imdb_data()
    elif dataset_name == "chinese":
        return load_chinese_data()
    elif dataset_name == "danish":
        return load_danish_data()
    elif dataset_name == "estonian":
        return load_estonian_data()
    elif dataset_name == "german":
        return load_german_data()
    elif dataset_name == "greek":
        return load_greek_data()
    elif dataset_name == "italian":
        return load_italian_data()
    elif dataset_name == "arabic":
        return load_arabic_data()
    elif dataset_name == "russian":
        return load_russian_data()
    elif dataset_name == "latvian":
        return load_latvian_data()
    elif dataset_name == "hindi":
        return load_hindi_data()
    elif dataset_name == "english":
        return load_stromfront_data()
    elif dataset_name == "portuguese":
        return load_portuguese_data()
    elif dataset_name == "ukraninan":
        return load_ukranian_data()
    elif dataset_name == "turkish":
        return load_turkish_data()
    elif dataset_name == "albanian":
        return load_albanian_data()
    elif dataset_name == "agnews":
        return load_agnews_data()
    elif dataset_name == "jigsaw_mola":
        return load_jigsaw_data()
    elif dataset_name == "ghc":
        return load_ghc_data()
    elif dataset_name == "ucc":
        return load_ucc_data()

def create_logger(save_path, log_level=logging.INFO):
    EXPERIMENT_DIRECTORY = save_path
# Configure the logging settings
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Create a log file with the current date and time in its name
    current_datetime = datetime.datetime.now()
    log_file = current_datetime.strftime("%Y-%m-%d_%H-%M-%S.log")

    # Create a file handler to write log messages to the specified log file
    file_handler = logging.FileHandler(os.path.join(EXPERIMENT_DIRECTORY, log_file))

    # Set the log level for the file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages (if you want a different format for the log file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)