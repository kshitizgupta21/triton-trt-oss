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
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)
# T5
from T5.T5ModelConfig import T5ModelTRTConfig, T5Metadata
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
from T5.export import T5DecoderONNXFile, T5EncoderONNXFile
from T5.trt import T5TRTEncoder, T5TRTDecoder

# NNDF
from NNDF.networks import NetworkMetadata, Precision
from NNDF.torch_utils import expand_inputs_for_beam_search

parser = argparse.ArgumentParser()
parser.add_argument("--fp16", help="whether to use FP16 for building TRT engine; false by default", action="store_true")
parser.add_argument("--preview-dynamic-feature", help="whether to use Preview 8.5 dynamic shapes feature; true by default", action="store_false")
parser.add_argument("--min-batch-size", type=int, help="min batch size for TRT engine, default 1", default=1)
parser.add_argument("--opt-batch-size", type=int, help="opt batch size for TRT engine, default 1", default=1)
parser.add_argument("--max-batch-size", type=int, help="max batch size for TRT engine, default 1", default=1)
parser.add_argument("--variant", type=str, help="variant of T5, default t5-small", default="t5-small")
parser.add_argument("--num-beams", type=int, help="beam size, by default set to 1 i.e. greedy search", default=1)
parser.add_argument("--trt-model-path", type=str, help="required; trt model save path, helpful to save in triton model repository", required=True) # e.g. ../../model_repository/trt_t5_bs1_beam/1/models/
args = parser.parse_args()

T5_VARIANT = args.variant

t5_model = T5ForConditionalGeneration.from_pretrained(T5_VARIANT) 
tokenizer = T5Tokenizer.from_pretrained(T5_VARIANT)
config = T5Config.from_pretrained(T5_VARIANT)

# save model locally
pytorch_model_dir = './models/{}/pytorch'.format(T5_VARIANT)
os.makedirs(pytorch_model_dir, exist_ok=True)

if os.path.exists(pytorch_model_dir) and len(os.listdir(pytorch_model_dir)) != 0:
    print('PyTorch model already exists. Skipping...')
else:
    t5_model.save_pretrained(pytorch_model_dir)
    print("PyTorch model saved to {}".format(pytorch_model_dir))
    
# convert to ONNX
onnx_model_path = './models/{}/ONNX'.format(T5_VARIANT)
os.makedirs(onnx_model_path, exist_ok=True)

metadata=NetworkMetadata(variant=T5_VARIANT, precision=Precision(fp16=args.fp16), other=T5Metadata(kv_cache=False))

encoder_onnx_model_fpath = T5_VARIANT + "-encoder.onnx"
decoder_onnx_model_fpath = T5_VARIANT + "-decoder-with-lm-head.onnx"

t5_encoder = T5EncoderTorchFile(t5_model.to('cpu'), metadata)
t5_decoder = T5DecoderTorchFile(t5_model.to('cpu'), metadata)

onnx_t5_encoder = t5_encoder.as_onnx_model(
    os.path.join(onnx_model_path, encoder_onnx_model_fpath), force_overwrite=False
)
onnx_t5_decoder = t5_decoder.as_onnx_model(
    os.path.join(onnx_model_path, decoder_onnx_model_fpath), force_overwrite=False
)

# convert ONNX to TensorRT
tensorrt_model_path = args.trt_model_path
os.makedirs(tensorrt_model_path, exist_ok=True)
# Decoder optimization profiles
max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[T5_VARIANT]
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
    min=(min_batch_size * num_beams, 1, max_sequence_length),
    opt=(opt_batch_size * num_beams, max_sequence_length // 2, max_sequence_length),
    max=(max_batch_size * num_beams, max_sequence_length, max_sequence_length),
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
print("Building TRT T5 Encoder Engine...")
t5_trt_encoder_engine = T5EncoderONNXFile(
                os.path.join(onnx_model_path, encoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", profiles=[encoder_profile], preview_features=preview_features)

print("-----------------------------------------------------------------")
print("Finished Building TRT T5 Encoder Engine")
print("-----------------------------------------------------------------")
print("Building TRT T5 Decoder Engine...")    
t5_trt_decoder_engine = T5DecoderONNXFile(
                os.path.join(onnx_model_path, decoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", profiles=[decoder_profile], preview_features=preview_features)

print("-----------------------------------------------------------------")
print("Finished Building TRT T5 Decoder Engine")
