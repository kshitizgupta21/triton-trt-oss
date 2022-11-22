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
from T5.T5ModelConfig import T5ModelTRTConfig, T5Metadata
from T5.trt import T5TRTEncoder, T5TRTDecoder
from T5.export import T5EncoderTRTEngine, T5DecoderTRTEngine
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

# from HuggingFace transformers
from transformers.generation_logits_process import (
    MinLengthLogitsProcessor,
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
T5_VARIANT = "t5-small"
num_beams = 2
early_stopping = False
max_length = 200
min_length = 30
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
        self.metadata = NetworkMetadata(variant=T5_VARIANT, precision=Precision(fp16=True), other=T5Metadata(kv_cache=use_cache))
        encoder_onnx_model_fpath = T5_VARIANT + "-encoder.onnx"
        decoder_onnx_model_fpath = T5_VARIANT + "-decoder-with-lm-head.onnx"
        tensorrt_model_path = "models/"
        cur_folder = Path(__file__).parent
        tensorrt_model_path = str(cur_folder / tensorrt_model_path)
        trt_config = AutoConfig.from_pretrained(T5_VARIANT)
        trt_config.use_cache = self.metadata.other.kv_cache
        trt_config.num_layers = T5ModelTRTConfig.NUMBER_OF_LAYERS[T5_VARIANT]
        # Initialize TensorRT engines from disk
        t5_trt_encoder_engine = T5EncoderTRTEngine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", self.metadata)
        t5_trt_decoder_engine = T5DecoderTRTEngine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", self.metadata)
        self.t5_trt_encoder = T5TRTEncoder(
                        t5_trt_encoder_engine, self.metadata, trt_config, batch_size=self.max_batch_size
                    )
        self.t5_trt_decoder = T5TRTDecoder(
                        t5_trt_decoder_engine, self.metadata, trt_config, batch_size=self.max_batch_size, num_beams=num_beams
                    )
            
        tokenizer = AutoTokenizer.from_pretrained(T5_VARIANT)
        if early_stopping:
            self.logits_processor = LogitsProcessorList([])
        else:
            self.logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
            ])
        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)]) 
        self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print("TensorRT T5 Model in Python Backend initialized")

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
                encoder_last_hidden_state = self.t5_trt_encoder(input_ids=input_ids)
                self.t5_trt_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
                if num_beams > 1:
                    encoder_last_hidden_state = expand_inputs_for_beam_search(encoder_last_hidden_state, expand_size=num_beams)
                decoder_input_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.int32, device="cuda")
                if num_beams > 1:
                    decoder_input_ids = expand_inputs_for_beam_search(decoder_input_ids, expand_size=num_beams)
                if num_beams == 1:
                    decoder_output = self.t5_trt_decoder.greedy_search(
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
                    decoder_output = self.t5_trt_decoder.beam_search(
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