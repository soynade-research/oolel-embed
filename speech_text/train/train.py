#!/usr/bin/env python
from ..models.hubert_embedding import HubertForEmbedding
from pathlib import Path
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_from_disk

import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    speech_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    text_model_path_or_name: str = field(
        default="soynade-research/oolel-embed",
        metadata={"help": "The sentence transformer model to use to embed the text."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    freeze_speech_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the speech encoder."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer."},
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."},
    )
    ctc_zero_infinity: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly"
            " occur when the inputs are too short to be aligned to the targets."
        },
    )
    add_adapter: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether a convolutional attention network should be stacked on top of the Wav2Vec2Bert Encoder. Can be very"
            "useful to downsample the output length."
        },
    )
    loss: Optional[str] = field(
        default="contrastive",
        metadata={"help": "The loss to use. Can be one of 'contrastive', or 'regression'."},
    )
    training_mode: Optional[str] = field(
        default="clip",
        metadata={"help": "The training mode. Can be one of 'clip', or 'joint'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        metadata={"help": "Path or name of the dataset (cf `load_dataset` method of the Datasets library)."}
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (cf `load_dataset` method of the Datasets library)."
        },
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    chars_to_ignore: Optional[list[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: list[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The target language that should be used be"
                " passed to the tokenizer for tokenization. Note that"
                " this is only relevant if the model classifies the"
                " input audio to a sequence of phoneme sequences."
            )
        },
    )


@dataclass
class DataCollatorHubertEmbedding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        # process speech queries
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
            )

        # we have input_ids when prompts are given in the instruction following approach    
        if "input_ids" in features[0]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            input_ids = self.tokenizer.pad(input_ids, return_tensors="pt", padding=self.padding, max_length=self.max_length_labels)
            batch["input_ids"] = input_ids["input_ids"]
            batch["input_ids_attention_mask"] = input_ids["attention_mask"]
        
        # process documents
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # replace padding with -100 to ignore loss correctly
        label_features = self.tokenizer.pad(label_features, return_tensors="pt", padding=self.padding, max_length=self.max_length_labels)
        batch["labels"] = label_features["input_ids"]
        batch["labels_attention_mask"] = label_features["attention_mask"]

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_from_disk(data_args.dataset_name)

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.text_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    # create model
    model = HubertForEmbedding.from_pretrained(
        model_args.speech_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        text_model_path_or_name=model_args.text_model_path_or_name,
        matryoshka_dims=[1024, 512, 256, 128],
        loss=model_args.loss,
        training_mode=model_args.training_mode,
    )
    model.load_embedding_model()
    if model_args.freeze_speech_encoder:
        model.freeze_base_model()
    print(model)

    processor = AutoProcessor.from_pretrained(
        model_args.speech_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != processor.feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name,
            datasets.features.Audio(sampling_rate=processor.feature_extractor.sampling_rate),
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * processor.feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * processor.feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = processor.feature_extractor.model_input_names[0]

    tokenizer_fn = model.text_embedding_model.tokenizer

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = processor.feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        labels = tokenizer_fn(batch[text_column_name])
        batch["labels"] = labels["input_ids"]
        batch["labels_attention_mask"] = labels["attention_mask"]

        # if "prompt" in batch:
        #     inputs_ids = tokenizer_fn(batch["prompt"])
        #     batch["input_ids"] = inputs_ids["input_ids"]
        #     batch["input_ids_attention_mask"] = inputs_ids["attention_mask"]
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        p = Path(data_args.dataset_name)
        vectorized_datasets_output = str(p.parent / (p.name + f"_vectorized_{text_column_name}"))
        if Path(vectorized_datasets_output).exists():
            vectorized_datasets = load_from_disk(vectorized_datasets_output)
            print(f"Vectorized dataset loaded from {str(vectorized_datasets_output)}")
        else:
            print(raw_datasets)
            raw_datasets = raw_datasets.filter(lambda x: x[text_column_name] is not None and x[text_column_name].strip() != "", num_proc=num_workers)
            print(raw_datasets)
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                num_proc=num_workers,
                desc="preprocess datasets",
            )

            def is_audio_in_length_range(length):
                return length > min_input_length and length < max_input_length

            # filter data that is shorter than min_input_length
            vectorized_datasets = vectorized_datasets.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )
            vectorized_datasets.save_to_disk(vectorized_datasets_output, num_proc=32)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    # Instantiate custom data collator
    data_collator = DataCollatorHubertEmbedding(
        processor=processor, tokenizer=model.text_embedding_model.tokenizer, padding="longest"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=None,
        processing_class=processor,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=bool(list(Path(training_args.output_dir).glob("checkpoint-*"))))
        Path(f"{training_args.output_dir}/hubert_mrl").mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.hubert.state_dict(), f"{training_args.output_dir}/hubert_mrl/pytorch_model.bin")
        trainer.model.config.save_pretrained(f"{training_args.output_dir}/hubert_mrl")
        processor.save_pretrained(f"{training_args.output_dir}/hubert_mrl")
        trainer.model.text_embedding_model.save(f"{training_args.output_dir}/oolel_mrl")
        metrics = train_result.metrics
        # metrics["matryoshka_dim_stats"] = ", ".join(f"dim={d}: {trainer.model.dim_stats[d] / trainer.model.total_samples}" for d in trainer.model.matryoshka_dims)
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.speech_model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()