# ComfyUI JoyCaption-Beta-GGUF Node

This project provides a node for ComfyUI to use the JoyCaption-Beta model in GGUF format for image captioning.

**Acknowledgments:**

This node is based on [fpgaminer/joycaption_comfyui](https://github.com/fpgaminer/joycaption_comfyui), with modifications to support the GGUF model format.

## Usage

### Installation

This node requires `llama-cpp-python` to be installed.

**Important:**

* Installing with `pip install llama-cpp-python` will only enable CPU inference.
* To utilize NVIDIA GPU acceleration, install with the following command:
    ```bash
    pip install llama-cpp-python --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu124](https://abetlen.github.io/llama-cpp-python/whl/cu124)
    ```
    *(Adjust `cu124` according to your CUDA version)*
* For non-NVIDIA GPUs or other installation methods, please refer to the official `llama-cpp-python` documentation:
    [https://llama-cpp-python.readthedocs.io/en/latest/](https://llama-cpp-python.readthedocs.io/en/latest/)

`llama-cpp-python` is not listed in `requirements.txt` to allow users to manually install the correct version with GPU support.

### Workflow Example

You can view an example workflow image at `assets/example.png`.

![Workflow Example](assets/example.png)

### Model Download and Placement

You need to download the JoyCaption-Beta GGUF model and the corresponding mmproj model.

1.  Download the models from the following Hugging Face repositories:
    * **Main Model (Recommended):** [concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf](https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main)
        * Download the relevant `joycaption-beta` model files and the `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf` file.
    * **Other Quantized Versions:** [mradermacher/llama-joycaption-beta-one-hf-llava-GGUF](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/tree/main)
    * **IQ Quantized Version (Theoretically higher quality, potentially slower on CPU):** [mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF/tree/main)

2.  Place the downloaded model files into the `models\llava_gguf\` folder within your ComfyUI installation directory.

### Video Tutorial

You can refer to the following Bilibili video tutorial for setup and usage:

[https://www.bilibili.com/video/BV1gRVgzYEUPeee/](https://www.bilibili.com/video/BV1gRVgzYEUPeee/)  (Video in production)