# Example for Hosting TensorRT OSS HuggingFace Models on Triton Inference Server

## To build the TensorRT (TRT) Engines

1. Build TRT 8.5 OSS container
```
bash build_trt_oss_docker.sh
```

2. Launch the container
```
bash run_trt_oss_docker.sh
```

3. Change Directory and Pip install HF demo requirements
```
cd demo/HuggingFace
pip install -r requirements.txt
```

4. Run [build_t5_trt.py](demo/HuggingFace/build_t5_trt.py) to build T5 TRT engines and [build_bart_trt.py](demo/HuggingFace/build_bart_trt.py) to build BART engines.
    * Run `python3 build_t5_trt.py --help` to see all options.
    * [gen_t5_bs1_beam2.sh](demo/HuggingFace/gen_t5_bs1_beam2.sh) is bash script that uses python script [build_t5_trt.py](demo/HuggingFace/build_t5_trt.py) to generate T5 engines with batch size 1 and beam size 2 for t5-small variant and saves the TRT T5 engines in [Triton Model Repository](triton_model_repository) 
    * [gen_bart_bs1_greedy.sh](demo/HuggingFace/build_bart_trt.py) uses python script [build_bart_trt.py](demo/HuggingFace/build_bart_trt.py) to generate BART engines with batch size 1 and greedy search for bart-base variant and saves the TRT BART engines in [Triton Model Repository](triton_model_repository)


## Triton Inference

Triton Model Repository is located at [model_repository](./triton_model_repository). Each model has `model.py` associated with it and [config.pbtxt](https://github.com/triton-inference-server/server/blob/main/docs/README.md#model-configuration) along with T5/BART TRT OSS code dependencies.

We showcase 2 models BART and T5 here. Currently, **TRT T5 supports both beam search and greedy search.** **TRT BART only supports greedy search currently.** 
* `trt_t5_bs1_beam2` = TRT T5 Max Batch Size 1 Model with Beam Search=2
* `trt_bart_bs1_greedy` = TRT BART Max Batch Size 1 Model with Greedy Search

Currently, **TensorRT engines for T5 and BART don't produce correct output for Batch Sizes > 1 (this bug is being worked on).** So we only show batch size = 1 example for T5 and BART here.

### Steps for Triton TRT Inference

1. Build Custom Triton container with TRT and other dependencies. Dockerfile is [docker/triton_trt.Dockerfile](docker/triton_trt.Dockerfile)
```
cd docker
bash build_triton_trt_docker.sh
cd ..
```

2. Launch custom Triton container
```
bash run_triton_trt_docker.sh
```

3. Launch JupyterLab at port 8888
```
bash start_jupyter.sh
```

4. Run through [1_triton_server.ipynb](triton_notebooks/1_triton_server.ipynb) to Launch Triton Server
5. Run through [2_triton_client.ipynb](triton_notebooks/2_triton_client.ipynb) to perform sample inference for T5 and BART TRT OSS HuggingFace models using Triton Server.


