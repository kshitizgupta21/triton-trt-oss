import argparse
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import torch
import tensorrt as trt
from tensorrt import PreviewFeature
from polygraphy.backend.trt import Profile
import numpy as np

# huggingface
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoConfig
)
# BART
from BART.BARTModelConfig import BARTModelTRTConfig, BARTMetadata
from BART.export import BARTEncoderTorchFile, BARTDecoderTorchFile
from BART.export import BARTDecoderONNXFile, BARTEncoderONNXFile
from BART.trt import BARTTRTEncoder, BARTTRTDecoder

# NNDF
from NNDF.networks import NetworkMetadata, Precision
from NNDF.torch_utils import expand_inputs_for_beam_search

parser = argparse.ArgumentParser()
parser.add_argument("--fp16", help="whether to use FP16 for building TRT engine; false by default", action="store_true")
parser.add_argument("--preview-dynamic-feature", help="whether to use Preview 8.5 dynamic shapes feature; true by default", action="store_false")
parser.add_argument("--min-batch-size", type=int, help="min batch size for TRT engine; default 1", default=1)
parser.add_argument("--opt-batch-size", type=int, help="opt batch size for TRT engine; default 1", default=1)
parser.add_argument("--max-batch-size", type=int, help="max batch size for TRT engine; default 1", default=1)
parser.add_argument("--variant", type=str, help="variant of BART; default facebook/bart-base", default="facebook/bart-base")
parser.add_argument("--num-beams", type=int, help="beam size, by default set to 1 i.e. greedy search", default=1)
parser.add_argument("--trt-model-path", type=str, help="required; trt model save path, helpful to save in triton model repository", required=True) # e.g. ../../triton_model_repository/trt_bart_bs1_greedy/1/models/
args = parser.parse_args()

BART_VARIANT = args.variant

# mbart variant can't be recognized by HF AutoClass yet
if "mbart" not in BART_VARIANT:    
    bart_model = BartForConditionalGeneration.from_pretrained(BART_VARIANT) 
    tokenizer = BartTokenizer.from_pretrained(BART_VARIANT)
else:
    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
    bart_model = MBartForConditionalGeneration.from_pretrained(BART_VARIANT)
    tokenizer = MBart50Tokenizer.from_pretrained(BART_VARIANT, src_lang="en_XX")
    
config = AutoConfig.from_pretrained(BART_VARIANT)
# remove facebook prefix
BART_VARIANT_NAME = BART_VARIANT.replace("facebook/", "")
# save model locally

pytorch_model_dir = './models/{}/pytorch'.format(BART_VARIANT_NAME)
os.makedirs(pytorch_model_dir, exist_ok=True)

if os.path.exists(pytorch_model_dir) and len(os.listdir(pytorch_model_dir)) != 0:
    print('PyTorch model already exists. Skipping...')
else:
    bart_model.save_pretrained(pytorch_model_dir)
    print("PyTorch model saved to {}".format(pytorch_model_dir))
    
# convert to ONNX
onnx_model_path = './models/{}/ONNX'.format(BART_VARIANT_NAME)
os.makedirs(onnx_model_path, exist_ok=True)

metadata=NetworkMetadata(variant=BART_VARIANT, precision=Precision(fp16=args.fp16), other=BARTMetadata(kv_cache=False))

encoder_onnx_model_fpath = BART_VARIANT_NAME + "-encoder.onnx"
decoder_onnx_model_fpath = BART_VARIANT_NAME + "-decoder-with-lm-head.onnx"

BART_encoder = BARTEncoderTorchFile(bart_model.to('cpu'), metadata)
BART_decoder = BARTDecoderTorchFile(bart_model.to('cpu'), metadata)

onnx_BART_encoder = BART_encoder.as_onnx_model(
    os.path.join(onnx_model_path, encoder_onnx_model_fpath), force_overwrite=False
)
onnx_BART_decoder = BART_decoder.as_onnx_model(
    os.path.join(onnx_model_path, decoder_onnx_model_fpath), force_overwrite=False
)

# convert ONNX to TensorRT
tensorrt_model_path = args.trt_model_path
os.makedirs(tensorrt_model_path, exist_ok=True)
# Decoder optimization profiles
max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[BART_VARIANT]
encoder_hidden_size = BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[BART_VARIANT]
min_batch_size = args.min_batch_size
opt_batch_size = args.opt_batch_size
max_batch_size = args.max_batch_size
num_beams = args.num_beams
preview_features = []
if args.preview_dynamic_feature:
    preview_features = [PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
decoder_profile = Profile()
decoder_profile.add(
    "input_ids",
    min=(min_batch_size * num_beams, 1),
    opt=(opt_batch_size * num_beams, max_sequence_length // 2),
    max=(max_batch_size * num_beams, max_sequence_length),
)
decoder_profile.add(
    "encoder_hidden_states",
    min=(min_batch_size * num_beams, 1, encoder_hidden_size),
    opt=(opt_batch_size * num_beams, max_sequence_length // 2, encoder_hidden_size),
    max=(max_batch_size * num_beams, max_sequence_length, encoder_hidden_size),
)

# Encoder optimization profiles
encoder_profile = Profile()
encoder_profile.add(
    "input_ids",
    min=(min_batch_size, 1),
    opt=(opt_batch_size, max_sequence_length // 2),
    max=(max_batch_size, max_sequence_length),
)

# convert to trt engine, 
print("Building TRT BART Encoder Engine...")
BART_trt_encoder_engine = BARTEncoderONNXFile(
                os.path.join(onnx_model_path, encoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", profiles=[encoder_profile], preview_features=preview_features)

print("-----------------------------------------------------------------")
print("Finished Building TRT BART Encoder Engine")
print("-----------------------------------------------------------------")
print("Building TRT BART Decoder Engine...")    
BART_trt_decoder_engine = BARTDecoderONNXFile(
                os.path.join(onnx_model_path, decoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", profiles=[decoder_profile], preview_features=preview_features)

print("-----------------------------------------------------------------")
print("Finished Building TRT BART Decoder Engine")
