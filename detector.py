# detector.py

import torch
import numpy as np
from PIL import Image
import supervision as sv

def run_detection_on_image(
    image_rgb, text_prompts, dino_processor, dino_model, sam_predictor, device
):
    """
    Runs your specific multi-prompt DINO + SAM detection pipeline on a single image.

    Args:
        image_rgb (np.ndarray): The input image in RGB format.
        text_prompts (list[str]): A list of text prompts for detection (e.g., ["car", "bank"]).
        dino_processor: The loaded Grounding DINO processor.
        dino_model: The loaded Grounding DINO model.
        sam_predictor: The loaded SAM predictor.
        device (torch.device): The device to run models on.

    Returns:
        tuple: A tuple containing (supervision.Detections, list[str]) for the found objects.
               Returns (None, None) if no objects are found.
    """
    BOX_THRESHOLD = 0.35
    image_pil = Image.fromarray(image_rgb)
    
    # --- 1. Your multi-prompt Grounding DINO Logic ---
    all_boxes = []
    all_scores = []
    all_labels = []

    for prompt in text_prompts:
        inputs = dino_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=BOX_THRESHOLD
        )

        if len(results[0]["boxes"]) > 0:
            all_boxes.append(results[0]["boxes"])
            all_scores.append(results[0]["scores"])
            all_labels.extend(results[0]["text_labels"])

    if not all_boxes:
        return None, None

    # Concatenate results from all prompts
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    text_labels = all_labels

    # --- 2. SAM v1 Segmentation (with loop) ---
    sam_predictor.set_image(image_rgb)
    input_boxes_np = boxes.cpu().numpy()
    
    segmented_masks = []
    for box in input_boxes_np:
        masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
        segmented_masks.append(masks[0])

    if not segmented_masks:
        return None, None
        
    final_masks = np.stack(segmented_masks, axis=0)
    
    # --- 3. Create Detections Object ---
    # Note: Your annotators use sv.ColorLookup.INDEX, so class_id is not needed here.
    detections = sv.Detections(
        xyxy=input_boxes_np,
        confidence=scores.cpu().numpy(),
        mask=final_masks
    )
    
    return detections, text_labels