# Text-Guided Static Object Detection in 2D and Panoramic Videos

- **Name:** Chandra Prakash
- **Institute:** IIT Madras
- **ROLL:** EE22B111

This project is a proof-of-concept system for performing text-based object detection in both standard 2D videos and 360° panoramic videos. The system allows a user to specify one or more object classes via command-line prompts (e.g., "a car", "a person") and identifies, segments, and logs these objects in the video.

## Features

- **Command-Line Interface:** Flexible and scriptable execution using command-line arguments for video paths and text prompts.
- **Text-Based Search:** Uses Grounding DINO to detect objects in video frames based on natural language prompts.
- **Multi-Class Detection:** Can search for multiple distinct object classes in a single run.
- **Object Segmentation:** Utilizes the Segment Anything Model (SAM) to generate precise pixel-level masks for all detected objects.
- **360° Video Support:** Can process equirectangular (360°) videos by projecting them into multiple Normal Field of View (NFoV) images.
- **Dual Projection Methods:** Implements both a classic grid-based projection (py360convert) and a modern, differentiable projection (nvtorchcam).
- **VLM Prototype:** Includes a separate script to demonstrate how a Video Large Language Model (Qwen2.5-VL) can be used for high-level video summarization and question-answering.
- **Comprehensive Outputs:** Generates annotated frames for each projected view and a detailed .csv log of all detections.

## Project Structure

```
/mindflix
|
|-- /data/ # Place your input .mp4 videos here
| |-- vid01.mp4
| |-- vid3604.mp4
|
|-- /outputs/ # All generated outputs are saved here
| |-- /stage2_output/
| | |-- /frame_0000/
| | | |-- view_p0_y90_annotated.jpg
| | |-- ...
| | |-- results_360.csv
|
|-- /weights/ # Pre-trained model weights are stored here
| |-- sam_vit_h_4b8939.pth
| |-- groundingdino_swint_ogc.pth
|-- main.py # Main script for Stage 1 & 2 (Detection/Segmentation)
|-- detector.py # Module with core detection logic
|-- classicProjection.py # Module for classic 360° projection
|-- nvtorchcamProjection.py # Module for modern 360° projection
|-- vlm.py # Standalone script for Stage 3 (VLM search)
|-- requirements.txt # Project dependencies
|-- README.md # This file
```

## Setup and Installation

### Create and Activate a Virtual Environment:

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

## Install Dependencies:

The requirements.txt file contains all necessary packages.

```
pip install -r requirements.txt
```

## Download Pre-trained Model Weights:

You must manually download the weights for SAM and place them in the weights folders like the following.

```
 /weights/
| |-- sam_vit_h_4b8939.pth
| |-- groundingdino_swint_ogc.pth
```

### Links are provided below: 

- [Download groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- [Download sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# How to Run

## Stage 1 & 2: Object Detection and Segmentation

This pipeline finds and segments objects based on a list of prompts.

### Command Usage

The script takes the following arguments:

- `--video_path` **(required)**: Path to the input video file.
- `--prompts` **(required)**: A space-separated list of objects to detect. If a single prompt contains spaces (like `"a red car"`), enclose that specific prompt in double quotes.
- `--projection_method` *(optional)*: `classic` or `nvtorchcam`. Default is `classic`.
- `--output_dir` *(optional)*: Folder to save results in. Default is `outputs/stage2_output`.

### Examples

1. **Basic command with multiple single-word prompts (no quotes needed):**

```bash
python main.py --video_path data/vid3604 --prompts car tree building
```

## Stage 3: VLM-Based Search Prototype

### Natural Language Video QA Pipeline

This pipeline answers natural language questions about a video.
- The VLM used is **Qwen/Qwen2.5-VL-3B-Instruct**, a powerful vision-language model from Alibaba's Qwen series.

#### Command Usage

The script takes the following arguments:


- `--video_path` **(required)**: Path to the input 2D video file.  
- `--prompt` **(required)**: The question or instruction for the model. Must be enclosed in double quotes.  
- `--num_frames` *(optional)*: Number of frames to sample from the video. Default is `8`.


It will render as:

---

#### Example

This is the exact command to run the test case you specified:

```bash
python vlm.py --video_path data/vid01.mp4 --prompt "Give me a detailed summary of the action." --num_frames 16
```

# Note: 

If u properly add the path of the videos inside the data folder and run as it is commands you will get the outputs for the videos inside the data folder which I used and executed and got outputs. If u face errors of you dont get any outputs, you can navigate to the test folder where I have attached all the outputs I got when I was testing for all 3 stages. 

- Also, the NFOV projection using `nvTorchCam` is not giving good outputs.

## Implementation Notes & Assumptions

- **SAM Version:** I used sam verison 1 as I was facing errors with the existence of sam 2 and its usage. Link for SAM's model weight was given above. 

- **Modular Approach:** The project is built modularly. Core detection, 2D processing, 360° projections, and VLM logic are separated into different files for clarity and reusability.

- **Static Objects:** The detection system is designed for static or slow-moving objects, as it does not perform object tracking between frames.

- **Frame Sampling:** The detection pipeline processes one frame per second to balance performance and coverage. This assumes that target objects are visible for at least one second.

- **No 360° Result Merging:** For Stage 2, the system detects objects in each projected view and saves those views separately. It does not yet implement the complex coordinate transformation and Non-Maximal Suppression (NMS) required to merge these detections back onto a single panoramic frame. This is a key area for future improvement.

# Additional info: 

[DOCUMENT](https://docs.google.com/document/d/1VqUZ8fpdCLaWqSV4FnOOxu8jGjOijNw2xOtESeVnbcI/edit?usp=sharing)


