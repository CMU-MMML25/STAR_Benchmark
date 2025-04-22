import os
import torch
import clip
import av
import bisect
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

import os
import torch
import clip
import av
import bisect
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization_utils import read_video_pyav3

class CLIPKeyFrameExtractor:
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32", 
                 top_k: int = 3,
                 sample_rate: int = 4, 
                 similarity_threshold: float = None,
                 min_distance: int = 10):
        """
        Initialize the CLIP-based key frame extractor.

        Args:
            clip_model_name: The CLIP model variant to use
            top_k: number of key frames to extract
            sample_rate: sampling rate (FPS)
            similarity_threshold: minimum similarity threshold
            min_distance: minimum distance between frames
        """
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        self.top_k = top_k
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.min_distance = min_distance
        
        print(f"CLIP frame retriever initialized on {self.device} with {clip_model_name}")
        print(f"Parameters: top_k={top_k}, sample_rate={sample_rate}, min_distance={min_distance}")

    def encode_question(self, question: str) -> torch.Tensor:
        """
        Encode the question using CLIP's text encoder.

        Args:
            question: The question text

        Returns:
            The encoded question tensor
        """
        # Tokenize and encode the question
        with torch.no_grad():
            text_inputs = clip.tokenize([question]).to(self.device)
            question_features = self.model.encode_text(text_inputs)
            # Normalize the features
            question_features = question_features / question_features.norm(dim=-1, keepdim=True)

        return question_features

    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Encode the frames using CLIP's image encoder.

        Args:
            frames: Array of frames as numpy arrays

        Returns:
            The encoded frame tensors
        """
        frame_features = []

        # Process frames in batches to avoid memory issues
        batch_size = 32
        num_frames = len(frames)

        for i in range(0, num_frames, batch_size):
            batch_frames = frames[i:i + batch_size]

            # Preprocess the batch of frames
            preprocessed_frames = torch.stack([
                self.preprocess(Image.fromarray(frame))
                for frame in batch_frames
            ]).to(self.device)

            # Encode the batch of frames
            with torch.no_grad():
                batch_features = self.model.encode_image(preprocessed_frames)
                # Normalize the features
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            frame_features.append(batch_features)

        # Concatenate all batches
        frame_features = torch.cat(frame_features, dim=0)

        return frame_features

    def compute_similarities(self, question_features: torch.Tensor, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarities between the question and all frames.

        Args:
            question_features: The encoded question
            frame_features: The encoded frames

        Returns:
            Tensor of similarity scores for each frame
        """
        # Compute cosine similarities
        similarities = (100.0 * question_features @ frame_features.T).squeeze()

        return similarities

    def select_key_frames(
        self,
        similarities: torch.Tensor,
        frames: np.ndarray,
        timestamps: List[float]
    ) -> Dict:
        """
        Select the top-k most relevant frames based on similarity scores.

        Args:
            similarities: Tensor of similarity scores
            frames: Array of all extracted frames
            timestamps: List of frame timestamps
            top_k: Number of key frames to select
            similarity_threshold: Minimum similarity score required (optional)
            min_distance: Minimum distance between selected frames in terms of indices

        Returns:
            Dictionary containing selected frames, timestamps, and scores
        """
        top_k = self.top_k
        similarity_threshold = self.similarity_threshold
        min_distance = self.min_distance
                
        # Convert similarities to numpy array
        similarities_np = similarities.cpu().numpy()

        # Apply threshold if provided
        if similarity_threshold is not None:
            valid_indices = np.where(similarities_np >= similarity_threshold)[0]
            if len(valid_indices) == 0:
                print(f"No frames with similarity >= {similarity_threshold} found.")
                # Fall back to top-k without threshold
                valid_indices = np.arange(len(similarities_np))
        else:
            valid_indices = np.arange(len(similarities_np))

        # Get top-k indices with minimum distance constraint
        selected_indices = []

        # Sort valid indices by similarity score (descending)
        sorted_indices = valid_indices[np.argsort(-similarities_np[valid_indices])]

        for idx in sorted_indices:
            # Check if the current index is far enough from already selected indices
            if all(abs(idx - selected_idx) >= min_distance for selected_idx in selected_indices) or not selected_indices:
                selected_indices.append(idx)

            # Stop when we have enough frames
            if len(selected_indices) >= top_k:
                break

        # If we couldn't find enough frames with the distance constraint, relax it
        if len(selected_indices) < top_k and min_distance > 0:
            print(f"Could only find {len(selected_indices)} frames with min_distance={min_distance}.")
            print("Relaxing distance constraint...")
            remaining = top_k - len(selected_indices)

            # Consider indices not yet selected
            remaining_indices = [idx for idx in sorted_indices if idx not in selected_indices]
            selected_indices.extend(remaining_indices[:remaining])

        # Sort selected indices by their position in the video
        selected_indices.sort()

        # Gather the selected frames, timestamps, and scores
        key_frames = frames[selected_indices]
        key_timestamps = [timestamps[idx] for idx in selected_indices]
        key_scores = [similarities_np[idx] for idx in selected_indices]

        return {
            "frames": key_frames,
            "timestamps": key_timestamps,
            "scores": key_scores,
            "indices": selected_indices,
        }

    def visualize_key_frames(self, key_frames_data: Dict, question: str, output_path: str = None):
        """
        Visualize the selected key frames with their similarity scores.

        Args:
            key_frames_data: Dictionary containing key frames data
            question: The original question
            output_path: Path to save the visualization (optional)
        """
        key_frames = key_frames_data["frames"]
        scores = key_frames_data["scores"]
        timestamps = key_frames_data["timestamps"]

        n_frames = len(key_frames)

        # Set up the figure
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 4, 5))
        if n_frames == 1:
            axes = [axes]

        # Set the figure title to the question
        fig.suptitle(f"Question: {question}", fontsize=16)

        # Plot each key frame
        for i, (frame, score, timestamp) in enumerate(zip(key_frames, scores, timestamps)):
            axes[i].imshow(frame)
            axes[i].set_title(f"Score: {score:.2f}\nTime: {timestamp:.2f}s")
            axes[i].axis("off")

        plt.tight_layout()

        # Save the figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")

        plt.show()

    def extract_key_frames(
        self, 
        video_path: str, 
        question: str, 
        start: float = 0.0, 
        end: float = None, 
        visualize: bool = False, 
        output_dir: str = None
    ) -> Dict:
        """
        Extract key frames from a video that are most relevant to the given question.
    
        Args:
            video_path: Path to the video file
            question: The question about the video
            sample_rate: Sample frames at this FPS rate
            top_k: Number of key frames to extract
            similarity_threshold: Minimum similarity score required
            min_distance: Minimum distance between selected frames
            start: Start time in seconds
            end: End time in seconds
            visualize: Whether to visualize the results
            output_dir: Directory to save the extracted frames
    
        Returns:
            Dictionary containing key frames data
        """
        frames, indices, timestamps = read_video_pyav3(
            video_path, 
            start=start, 
            end=end, 
            sampling_fps=self.sample_rate
        )
    
        # Encode the question
        question_features = self.encode_question(question)
    
        # Encode the frames
        frame_features = self.encode_frames(frames)
    
        # Compute similarities between question and frames
        similarities = self.compute_similarities(question_features, frame_features)
    
        # Select key frames based on similarities
        key_frames_data = self.select_key_frames(
            similarities, 
            frames, 
            timestamps
        )
    
        # Save the key frames if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(key_frames_data["frames"]):
                frame_path = os.path.join(output_dir, f"{i+1}.png")
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_bgr)
                
            # Add the frame paths to the output data
            key_frames_data["frame_paths"] = [
                os.path.join(output_dir, f"{i+1}.png") 
                for i in range(len(key_frames_data["frames"]))
            ]
    
        # Visualize the results if requested
        if visualize:
            visualization_path = os.path.join(output_dir, "visualization.png") if output_dir else None
            self.visualize_key_frames(key_frames_data, question, visualization_path)
    
        return key_frames_data