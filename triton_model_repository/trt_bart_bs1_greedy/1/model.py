import numpy as np
import torch
import json
from pathlib import Path
import os
import sys
ROOT_DIR = os.path.abspath("./")
from pathlib import Path
sys.path.append(ROOT_DIR)
from NNDF.networks import NetworkMetadata, Precision
from NNDF.torch_utils import expand_inputs_for_beam_search
from BART.BARTModelConfig import BARTModelTRTConfig, BARTMetadata
from BART.trt import BARTTRTEncoder, BARTTRTDecoder
from BART.export import BARTEncoderTRTEngine, BARTDecoderTRTEngine
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

# from HuggingFace transformers
from transformers.generation_logits_process import (
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_beam_search import (
    BeamSearchScorer,
)
from transformers import AutoTokenizer, AutoConfig

# settings
BART_VARIANT = "facebook/bart-base"
BART_VARIANT_NAME = BART_VARIANT.replace("facebook/", "")
num_beams = 1
early_stopping = False
max_length = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[BART_VARIANT]
min_length = BARTModelTRTConfig.MIN_OUTPUT_LENGTH[BART_VARIANT]
# TRT KV Cache disabled due to performance improvements in progress, not beating non-KV version yet
use_cache = False


class TritonPythonModel:
    # Every Python model must have "TritonPythonModel" as the class name!
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        
        # Read Max Batch Size from Triton Model config
        model_config = json.loads(args["model_config"])
        self.max_batch_size = model_config["max_batch_size"]

        self.metadata = NetworkMetadata(variant=BART_VARIANT, precision=Precision(fp16=True), other=BARTMetadata(kv_cache=use_cache))

        encoder_onnx_model_fpath = BART_VARIANT_NAME + "-encoder.onnx"
        decoder_onnx_model_fpath = BART_VARIANT_NAME + "-decoder-with-lm-head.onnx"
        tensorrt_model_path = "models/"
        cur_folder = Path(__file__).parent
        tensorrt_model_path = str(cur_folder / tensorrt_model_path)
        trt_config = AutoConfig.from_pretrained(BART_VARIANT)
        trt_config.use_cache = self.metadata.other.kv_cache
        trt_config.num_layers = BARTModelTRTConfig.NUMBER_OF_LAYERS[BART_VARIANT]
        # Initialize TensorRT engines from disk
        BART_trt_encoder_engine = BARTEncoderTRTEngine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", self.metadata)
        BART_trt_decoder_engine = BARTDecoderTRTEngine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", self.metadata)
        self.BART_trt_encoder = BARTTRTEncoder(
                        BART_trt_encoder_engine, self.metadata, trt_config, batch_size=self.max_batch_size
                    )
        self.BART_trt_decoder = BARTTRTDecoder(
                        BART_trt_decoder_engine, self.metadata, trt_config, batch_size=self.max_batch_size, num_beams=num_beams
                    )
            
        tokenizer = AutoTokenizer.from_pretrained(BART_VARIANT)
        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)]) 
        no_repeat_ngram_size = BARTModelTRTConfig.NO_REPEAT_NGRAM_SIZE
        
        self.logits_processor = LogitsProcessorList([
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size), 
            MinLengthLogitsProcessor(min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)),
            ForcedBOSTokenLogitsProcessor(tokenizer.convert_tokens_to_ids(tokenizer.bos_token)),
            ForcedEOSTokenLogitsProcessor(max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
        ])
        self.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        print("TensorRT BART Model in Python Backend initialized")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
            
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            # use dlpack to convert Python backend tensor to PyTorch tensor without making any copies & then move to GPU
            input_ids = from_dlpack(input_ids.to_dlpack()).cuda()
            batch_size = input_ids.shape[0]
            with torch.no_grad():
                encoder_last_hidden_state = self.BART_trt_encoder(input_ids=input_ids)
                self.BART_trt_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
                if num_beams > 1:
                    encoder_last_hidden_state = expand_inputs_for_beam_search(encoder_last_hidden_state, expand_size=num_beams)
                decoder_input_ids = torch.full((batch_size, 1), self.eos_token_id, dtype=torch.int32, device="cuda")
                if num_beams > 1:
                    decoder_input_ids = expand_inputs_for_beam_search(decoder_input_ids, expand_size=num_beams)
                if num_beams == 1:
                    decoder_output = self.BART_trt_decoder.greedy_search(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_last_hidden_state,
                        stopping_criteria=self.stopping_criteria,
                        logits_processor=self.logits_processor,
                    )
                else:
                    beam_scorer = BeamSearchScorer(
                        batch_size=batch_size,
                        num_beams=num_beams,
                        device="cuda",
                        do_early_stopping=early_stopping,
                    )
                    decoder_output = self.BART_trt_decoder.beam_search(
                        input_ids=decoder_input_ids,
                        beam_scorer=beam_scorer,
                        encoder_hidden_states=encoder_last_hidden_state,
                        stopping_criteria=self.stopping_criteria,
                        logits_processor=self.logits_processor,
                        use_cache=self.metadata.other.kv_cache
                    )
            # torch tensor, move to CPU from GPU
            output_ids = decoder_output.cpu()
            # convert torch tensor to Python backend tensor in zero-copy fashion
            output_ids = pb_utils.Tensor.from_dlpack("output_ids", to_dlpack(output_ids))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_ids])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')