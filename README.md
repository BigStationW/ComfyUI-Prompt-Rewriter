# ComfyUI Prompt Manager

This is a custom fork of [this repository](https://github.com/FranckyB/ComfyUI-Prompt-Manager) incorporating [some enhancements](https://github.com/FranckyB/ComfyUI-Prompt-Manager/pull/3).

## Use case
This node can rewrite your prompts with the help of a chosen instruct (or thinking) LLM (GGUF).

<img width="700" alt="FINAL" src="https://github.com/user-attachments/assets/672fa21f-670d-4639-9121-5cb66ae4959b" />

## Installation

1) Navigate to the **ComfyUI/custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```
git clone https://github.com/BigStationW/ComfyUI-Prompt-Manager
```
2) Navigate to the **ComfyUI\custom_nodes\ComfyUI-Prompt-Manager** folder, open cmd and run:

```
..\..\..\python_embeded\python.exe -s -m pip install -r "requirements.txt"
```
3) If you have Windows, open cmd and run:
```
winget install llama.cpp
```
If you have another OS, you can refer to [this](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

To update llama.cpp, open cmd and run:
```
winget upgrade llama.cpp
```

## Instruct/Thinking LLMs

1) Navigate to the **ComfyUI\models** folder and create a folder named "gguf"
2) Navigate to the **ComfyUI\models\gguf** folder and place your chosen GGUF LLM file there.

For example you can go for this (Instruct model):
- https://huggingface.co/Qwen/Qwen3-4B-GGUF

Or something like this (thinking model):
-  https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF

You can even go for uncensored LLMs, like that one for example:
- https://huggingface.co/mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF

## Usage
It should look like this.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/8095c9d6-15ce-4680-9dc3-7fe107973f9e" />

An example workflow (for Z-image turbo) can be found [here](https://github.com/BigStationW/ComfyUI-Prompt-Manager/blob/main/workflow/workflow_Z-image_turbo.json).

PS: **The Display Any (rgthree)** node can be found [here](https://github.com/rgthree/rgthree-comfy).

## MultiGPU and offloading

This node allows you to split the model into your GPUs and the CPU with the **gpu_layer**s placeholder.

For example:
- Empty placeholder -> All layers go to the first GPU (default)
- gpu0:0.7 -> 70% to GPU:0, 30% to CPU
- gpu0:0.5, gpu1:0.4 -> 50% GPU:0, 40% GPU:1, 10% CPU

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d5869277-8c3e-439f-a08a-5a0401835d6f" />


## Image inputs
For visual models, you can add up to 5 images to the Prompt Generator Options node.

1. Download a visual model and put it to the **ComfyUI\models\gguf** folder, like that one for example:

- https://huggingface.co/unsloth/Qwen3-VL-4B-Thinking-GGUF

2. Download its mmproj file and put it to the **ComfyUI\models\gguf** folder.
- https://huggingface.co/unsloth/Qwen3-VL-4B-Thinking-GGUF/blob/main/mmproj-BF16.gguf

3. You have to rename "mmproj-BF16.gguf" to "Qwen3-VL-4B-Thinking-mmproj-BF16.gguf"

At the end it'll look like this:

<img width="700" alt="image" src="https://github.com/user-attachments/assets/4cdb8201-afac-4360-b1c0-b9de26e87c87" />



