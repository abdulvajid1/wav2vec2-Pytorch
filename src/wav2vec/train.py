import os
import shutil
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from .dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPretraining
from .model import Wav2Vec2ForPreTraining
from .utils import Wav2Vec2Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 model")

    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str,
        help="Name of the experiment. This will be used to create a directory to save checkpoints"
    )

    parser.add_argument(
        "--working_dir",
        required=True,
        type=str,
        help="Working directory where the experiment directory will be created"
    )

    parser.add_argument(
        "--path_to_data_root",
        required=True,
        type=str,
        help="Path to the root directory of the dataset"
    )

    parser.add_argument(
        "--train_split",
        help="Name of the training split. For example, 'train-clean-100' for LibriSpeech dataset",
        required=True,
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        choices=["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other"],
        nargs="+",
        type=str,
    )

    parser.add_argument(
        "--test_split",
        help="Name of the test split. For example, 'test-clean' for LibriSpeech dataset",
        required=True,
        default=["test-clean", "test-other"],
        choices=["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other"],
        nargs="+",
        type=str,
    )

    parser.add_argument(
        "--min_duration_in_seconds",
        help="Minimum duration of audio files in seconds. Audio files shorter than this will be filtered out",
        default=2.0,
        type=float,
    )

    parser.add_argument(
        "--max_duration_in_seconds",
        help="Maximum duration of audio files in seconds. Audio files longer than this will be filtered out",
        default=20.0,
        type=float,
    )

    parser.add_argument(
        "--sample_rate",
        help="Sample rate of the audio files. All audio files will be resampled to this sample rate",
        default=16000,
        type=int,
    )

    parser.add_argument(
        "--audio_input_channels",
        help="Number of input channels for the audio files",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--masking_prob",
        help="Probability of masking each token in the input",
        default=0.065,
        type=float,
    )

    parser.add_argument(
        "--masking_span_length",
        help="Length of the mask to be applied to the input",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--num_negatives",
        help="Number of negative samples to be used for contrastive loss",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers to be used for data loading",
        default=4,
        type=int,
    )   

    parser.add_argument(
        "--conv_dim",
        help="Dimension of the convolutional layers in the feature extractor",
        default=(512, 512, 512, 512, 512, 512, 512),
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--conv_kernel",
        help="Kernel size of the convolutional layers in the feature extractor",
        default=(10, 3, 3, 3, 3, 2, 2),
        nargs="+",
        type=int,
    )   

    parser.add_argument(
        "--conv_stride",
        help="Stride of the convolutional layers in the feature extractor",
        default=(5, 2, 2, 2, 2, 2, 2),
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--disable_conv_bias",
        help="Whether to disable bias in the convolutional layers",
        action=argsparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--feature_projection_dropout",
        help="Dropout probability for the feature projection layer",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--conv_positional_embedding_dropout",
        help="Dropout probability for the convolutional positional embedding layer",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--conv_positional_emb_groups",
        help="Number of groups for the convolutional positional embedding layer",
        default=16,
        type=int,
    )


    parser.add_argument(
        "--conv_positional_emb_kernel",
        help="Kernel size for the convolutional positional embedding layer",
        default=128,
        type=int,
    )   

    parser.add_argument(
        "--num_transformer_layers",
        help="Number of transformer layers in the model",
        default=12,
        type=int,
    )

    parser.add_argument(
        "--mlp_ratio",
        help="Ratio of the hidden dimension to the input dimension in the MLP layers of the transformer",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--mlp_dropout",
        help="Dropout probability for the MLP layers of the transformer",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--attention_dropout",
        help="Dropout probability for the attention layers of the transformer",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--transformer_encoder_dropout",
        help="Dropout probability for the transformer encoder layers",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--layer_dropout",
        help="Dropout probability for the transformer layers",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--initializer_range",
        help="Standard deviation of the truncated_normal_initializer for initializing all weight matrices",
        default=0.02,
        type=float,
    )

    parser.add_argument(
        "--num_codevector_groups",
        help="Number of groups for the codevector layer",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--num_codevectors_per_group",
        help="Number of codevectors per group for the codevector layer",
        default=320,
        type=int,
    )

    parser.add_argument(
        "--codevector_dim",
        help="Dimension of the codevectors for the codevector layer",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--pre_quantizer_dropout",
        help="Dropout probability for the pre-quantizer layer",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--max_gumble_temperature",
        help="Maximum temperature for the Gumbel softmax in the quantizer",
        default=2.0,
        type=float,
    )

    parser.add_argument(
        "--min_gumble_temperature",
        help="Minimum temperature for the Gumbel softmax in the quantizer",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--gumbel_temperature_decay",
        help="Decay rate for the Gumbel softmax temperature in the quantizer",
        default=0.999995,
        type=float,
    )

    parser.add_argument(
        "--diversity_loss_weight",
        help="Weight for the diversity loss in the total loss",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--contrastive_logits_temperature",
        help="Temperature for the contrastive logits in the contrastive loss",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--per_gpu_batch_size",
        help="Batch size per GPU",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--grad_accumulation_steps",
        help="Number of gradient accumulation steps",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--num_training_steps",
        help="Total number of training steps",
        default=200000,
        type=int,
    )

    parser.add_argument(
        "--num_warmup_steps",
        help="Number of warmup steps for the learning rate scheduler",
        default=10000,
        type=int,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        help="Type of learning rate scheduler to use",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        type=str,
    )

    parser.add_argument(
        "--logging_steps",
        help="Number of steps between logging training metrics",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--checkpoint_interval",
        help="Number of steps between saving model checkpoints",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--eval_interval",
        help="Number of steps between evaluating the model on the test set",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--learning_rate",
        help="Learning rate for the optimizer",
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--bias_weight_decay",
        help="Weight decay for the bias parameters",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--norm_weight_decay",
        help="Weight decay for the normalization parameters",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay for the optimizer",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "--adam_beta1",
        help="Beta1 parameter for the Adam optimizer",
        default=0.9,
        type=float,
    )

    parser.add_argument(
        "--adam_beta2",
        help="Beta2 parameter for the Adam optimizer",
        default=0.98,
        type=float,
    )

    parser.add_argument(
        "--adam_epsilon",
        help="Epsilon parameter for the Adam optimizer",
        default=1e-6,
        type=float,
    )


    parser.add_argument(
        "--num_keep_checkpoints",
        help="Number of checkpoints to keep. Older checkpoints will be deleted",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        help="Path to a checkpoint to resume training from",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--log_wandb",
        help="Whether to log training metrics to Weights & Biases",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    return args



# HELPER Function
def multiply_gradients(params, constant):
    for param in params:
        if param.grad is not None:
            param.grad.data.mul_(constant)


def compute_gradient_norms(params, scale=1):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm() ** 2

    total_norm = total_norm ** 0.5
    return total_norm


def compute_batch_duration(attention_mask, sampling_rate):
    total_duration_in_seconds = torch.sum(attention_mask.sum(axis=-1)/ sampling_rate)
    total_duration_in_houres = total_duration_in_seconds / 3600
    return total_duration_in_houres


args = parse_args()

if args.seed is not None:
    set_seed(args.seed)

