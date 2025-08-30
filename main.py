# main.py (Final Version with Command-Line Arguments)

import cv2
import os
import torch
import pandas as pd
import supervision as sv
import argparse  # --- ADDED: For command-line arguments ---

# Import our custom modules
from classicProjection import project_to_nfov_grid
from nvtorchcamProjection import project_to_nfov_nvtorchcam
from detector import run_detection_on_image

# --- Model Loading (from segment_anything and transformers) ---
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from segment_anything import SamPredictor, sam_model_registry

# --- REMOVED: Hardcoded configuration is now handled by argparse ---

# --- Global Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("Loading all models, this may take a moment...")

# Load Grounding DINO
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
dino_processor = GroundingDinoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = GroundingDinoForObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)

# Load SAM v1
SAM_MODEL_TYPE = "vit_h" 
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam_model)

print("All models loaded successfully.")

# --- Create Annotators ---
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

# --- Main 360 Video Processing Function ---
# --- MODIFIED: Function now accepts configuration as arguments ---
def process_360_video(video_path, text_prompts, projection_method, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = int(fps) if fps > 0 else 30
    frame_count = 0
    
    all_csv_data = []

    print(f"\nProcessing 360 video: {video_path} with prompts: {', '.join(text_prompts)}")

    while True:
        ret, panoramic_frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            timestamp_sec = frame_count / fps if fps > 0 else frame_count / 30.0
            print(f"\n--- Processing Panoramic Frame {frame_count} at {timestamp_sec:.2f}s ---")

            panoramic_frame_rgb = cv2.cvtColor(panoramic_frame, cv2.COLOR_BGR2RGB)

            # --- 1. Project to NFoV Views ---
            if projection_method == 'classic':
                nfov_images, angles = project_to_nfov_grid(panoramic_frame_rgb)
            else:
                nfov_images, angles = project_to_nfov_nvtorchcam(panoramic_frame_rgb)
            
            frame_output_dir = os.path.join(output_dir, f"frame_{frame_count:04d}")
            os.makedirs(frame_output_dir, exist_ok=True)

            # --- 2. Run Detection on Each View ---
            for i, nfov_image in enumerate(nfov_images):
                pitch, yaw = angles[i]
                print(f"  - Running detection on view (Pitch: {pitch}, Yaw: {yaw})")
                
                detections, labels = run_detection_on_image(
                    nfov_image, text_prompts, dino_processor, dino_model, sam_predictor, DEVICE
                )
                
                if detections:
                    print(f"    > Found {len(detections)} objects. Saving outputs...")
                    
                    nfov_bgr = cv2.cvtColor(nfov_image, cv2.COLOR_RGB2BGR)
                    detection_labels = [f"{labels[i]} {detections.confidence[i]:0.2f}" for i in range(len(labels))]
                    
                    annotated_nfov = mask_annotator.annotate(scene=nfov_bgr.copy(), detections=detections)
                    annotated_nfov = box_annotator.annotate(scene=annotated_nfov, detections=detections)
                    annotated_nfov = label_annotator.annotate(scene=annotated_nfov, detections=detections, labels=detection_labels)

                    cv2.imwrite(os.path.join(frame_output_dir, f"view_p{pitch}_y{yaw}_annotated.jpg"), annotated_nfov)

                    for j in range(len(detections)):
                        box = detections.xyxy[j]
                        all_csv_data.append({
                            'timestamp': f"{timestamp_sec:.2f}",
                            'pano_frame': frame_count, 'view_pitch': pitch, 'view_yaw': yaw,
                            'object': labels[j],
                            'x1': int(box[0]), 'y1': int(box[1]),
                            'x2': int(box[2]), 'y2': int(box[3])
                        })

            print(f"Finished processing all views for frame {frame_count}.")
            
        frame_count += 1
    
    cap.release()

    if all_csv_data:
        csv_path = os.path.join(output_dir, "results_360.csv")
        df = pd.DataFrame(all_csv_data)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved all detections to {csv_path}")
    else:
        print("\nNo detections were made across any views, so no CSV file was created.")

    print("\n\n360 video processing complete.")

if __name__ == "__main__":
    # --- ADDED: Setup for command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run text-based object detection on 2D or 360 videos.")
    
    parser.add_argument(
        "--video_path", 
        type=str, 
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--prompts", 
        nargs='+',  # This allows for one or more prompt arguments
        required=True,
        help="A list of object prompts to detect (e.g., --prompts person car bicycle)."
    )
    parser.add_argument(
        "--projection_method", 
        type=str, 
        default='classic',
        choices=['classic', 'nvtorchcam'],
        help="The projection method to use for 360 videos."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='outputs/stage2_output',
        help="The directory where output files will be saved."
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
        # --- MODIFIED: Call the main function with the parsed arguments ---
        process_360_video(
            video_path=args.video_path,
            text_prompts=args.prompts,
            projection_method=args.projection_method,
            output_dir=args.output_dir
        )