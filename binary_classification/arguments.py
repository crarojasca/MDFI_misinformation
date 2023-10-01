from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_type: str = field(
        metadata={"help": "Task type, can be either generation or classification"}
    )
    num_labels: str = field(
        metadata={"help": "Number of labels, used for sequence classification"}
    )
    mode: str = field(
        metadata={"help": "mode, can be either train or test"}
    )
    problem_type: str = field(
        metadata={"help": " Can be one of \"regression\", \"single_label_classification\" or \"multi_label_classification\""}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    save_name: str = field(default="best_epoch", metadata={"help":"Model saved under this name"})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_type: Optional[str] = field(
        default="test", metadata={"help": "The type_path of the test file, test.seen, test.unseen etc."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=None, metadata={"help": "# training examples. None means use all."})
    n_val: Optional[int] = field(default=None, metadata={"help": "# validation examples. None means use all."})
    n_test: Optional[int] = field(default=None, metadata={"help": "# test examples. None means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


@dataclass
class EvalArguments:
    """
    Arguments pertaining to the evaluation of the model.
    """

    decode: Optional[str] = field(
        default='beam_search', metadata={"help": "Decoding method used, take in value of beam_search, nucleus"}
    )
    metric: Optional[str] = field(
        default='bleu', metadata={"help": "The metric used to evaluate the model, takes in value of bleu, rouge, meteor etc"}
    )
    compute_metric: Optional[bool] = field(
        default=False, metadata={"help": "whether to compute metrics while generating the outputs, must be False if num_samples > 1"}
    )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": "beam size used to decode"}
    )
    num_samples: Optional[int] = field(
        default=1, metadata={"help": "Number of decoded sequence for each input"}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    report_to: Optional[int] = field(
        default="wandb",
        metadata={"help": "Where the data will be logged"}
    )
