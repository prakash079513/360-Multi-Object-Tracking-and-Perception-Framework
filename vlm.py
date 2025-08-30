# vlm.py (Final Version with Command-Line Arguments)

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import argparse  # --- ADDED: For command-line arguments ---

# --- REMOVED: Hardcoded configuration is now handled by argparse ---

# --- Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"Loading model: {MODEL_ID}.")

# Use the fast processor for better performance
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
print("VLM model loaded successfully.")


def sample_frames_from_video(video_path, num_frames):
    """
    Samples a specified number of frames evenly from a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        indices = np.arange(total_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    print(f"Sampled {len(frames)} frames from the video.")
    return frames

def run_vlm_inference(text_prompt, video_frames):
    """
    Combines text and video frames into a prompt and gets a response from the VLM.
    """
    if not video_frames:
        print("Cannot run inference, no video frames were provided.")
        return

    # Create the chat structure with image PLACEHOLDERS
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": text_prompt})
    for _ in video_frames:
        messages[0]["content"].append({"type": "image"})

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt_text], images=video_frames, return_tensors="pt").to(DEVICE)

    print("\nModel is generating a response...")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    response_ids = output_ids[0][len(inputs.input_ids[0]):]
    response = processor.decode(response_ids, skip_special_tokens=True)

    return response

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- ADDED: Setup for command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Ask natural language questions about a video using a VLM.")
    
    parser.add_argument(
        "--video_path", 
        type=str, 
        required=True,
        help="Path to the input 2D video file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The natural language question or prompt to ask the model. Enclose in quotes if it contains spaces."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="The number of frames to sample from the video. Default is 8."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
        video_frames = sample_frames_from_video(args.video_path, args.num_frames)
        
        if video_frames:
            # --- MODIFIED: Use parsed arguments ---
            model_response = run_vlm_inference(args.prompt, video_frames)
            
            print("\n" + "="*30)
            print("      VLM PROTOTYPE RESULT")
            print("="*30)
            print(f"Your Prompt: {args.prompt}")
            print(f"Model's Response: {model_response}")
            print("="*30)