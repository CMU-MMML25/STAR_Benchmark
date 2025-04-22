from typing import List, Dict

import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

from qwen_vl_utils import process_vision_info


class CLIPKeyFrameExtractor:
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        """
        Initialize the CLIP-based key frame extractor.

        Args:
            clip_model_name: The CLIP model variant to use
        """
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        print(f"CLIP model loaded on {self.device}")

    def extract_frames(self, video_path: str, start: float = 0.0, end: float = None, sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from the video at the given sample rate within a specified time range.
    
        Args:
            video_path: Path to the video file
            start: Start timestamp in seconds (default: 0.0, i.e., from the beginning)
            end: End timestamp in seconds (default: None, i.e., until the end)
            sample_rate: Sample every nth frame (default: 1, i.e., extract all frames)
    
        Returns:
            List of extracted frames as numpy arrays, frame indices, and fps
        """
        frames = []
        frame_indices = []
    
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
    
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Calculate start and end frames
        start_frame = int(start * fps) if start is not None else 0
        end_frame = int(end * fps) if end is not None else total_frames
        
        print(f"Video has {total_frames} frames at {fps} FPS, duration: {duration:.2f}s")
        print(f"Extracting frames from {start:.2f}s to {end if end is not None else duration:.2f}s")
        print(f"Frame range: {start_frame} to {end_frame}")
        
        # Set the starting position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames within the specified range
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
    
            if (frame_idx - start_frame) % sample_rate == 0:
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_idx)
    
            frame_idx += 1
    
            # Print progress every 100 frames
            if (frame_idx - start_frame) % 100 == 0:
                progress = (frame_idx - start_frame) / (end_frame - start_frame)
                print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame} frames ({progress:.1%})")
    
        cap.release()
        print(f"Extracted {len(frames)} frames at sample rate {sample_rate}")
        return frames, frame_indices, fps
    
    def get_frame_timestamps(self, frame_indices: List[int], fps: float) -> List[float]:
        """
        Convert frame indices to timestamps in seconds.

        Args:
            frame_indices: List of frame indices
            fps: Frames per second of the video

        Returns:
            List of timestamps in seconds
        """
        return [idx / fps for idx in frame_indices]

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

    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Encode the frames using CLIP's image encoder.

        Args:
            frames: List of frames as numpy arrays

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

            print(f"Encoded frames {i} to {min(i + batch_size, num_frames)}/{num_frames}")

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
        frames: List[np.ndarray],
        frame_indices: List[int],
        timestamps: List[float],
        top_k: int = 5,
        similarity_threshold: float = None,
        min_distance: int = 0
    ) -> Dict:
        """
        Select the top-k most relevant frames based on similarity scores.

        Args:
            similarities: Tensor of similarity scores
            frames: List of all extracted frames
            frame_indices: List of frame indices
            timestamps: List of frame timestamps
            top_k: Number of key frames to select
            similarity_threshold: Minimum similarity score required (optional)
            min_distance: Minimum distance between selected frames in terms of indices

        Returns:
            Dictionary containing selected frames, indices, timestamps, and scores
        """
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

        # Gather the selected frames, their original indices, timestamps, and scores
        key_frames = [frames[idx] for idx in selected_indices]
        key_frame_indices = [frame_indices[idx] for idx in selected_indices]
        key_timestamps = [timestamps[idx] for idx in selected_indices]
        key_scores = [similarities_np[idx] for idx in selected_indices]

        return {
            "frames": key_frames,
            "indices": key_frame_indices,
            "timestamps": key_timestamps,
            "scores": key_scores
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

    def extract_key_frames(self, video_path: str, question: str, sample_rate: int = 1,
        top_k: int = 5, similarity_threshold: float = None, min_distance: int = 0,
        start: Optional[int] = None, end: Optional[int] = None, 
        visualize: Optional[bool] = False, output_dir: Optional[str] = None) -> Dict:
        """
        Extract key frames from a video that are most relevant to the given question.
    
        Args:
            video_path: Path to the video file
            question: The question about the video
            sample_rate: Sample every nth frame
            top_k: Number of key frames to extract
            similarity_threshold: Minimum similarity score required
            min_distance: Minimum distance between selected frames
            visualize: Whether to visualize the results
            output_dir: Directory to save the extracted frames
    
        Returns:
            Dictionary containing key frames data
        """
        print(f"Processing video: {video_path}")
        print(f"Question: {question}")
    
        # Extract frames from the video
        frames, frame_indices, fps = self.extract_frames(video_path, start, end, sample_rate)
        timestamps = self.get_frame_timestamps(frame_indices, fps)
    
        # Encode the question
        question_features = self.encode_question(question)
    
        # Encode the frames
        frame_features = self.encode_frames(frames)
    
        # Compute similarities between question and frames
        similarities = self.compute_similarities(question_features, frame_features)
    
        # Select key frames based on similarities
        key_frames_data = self.select_key_frames(
            similarities, frames, frame_indices, timestamps,
            top_k, similarity_threshold, min_distance
        )
    
        # Save the key frames if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(key_frames_data["frames"]):
                frame_path = os.path.join(output_dir, f"{i+1}.png")
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_bgr)
                print(f"Saved frame to {frame_path}")
            
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


def retrieve_frames(frame_retriever, query: str, video_path: str, start: float, end: float) -> List[Dict]:
    """
    Retrieve relevant frames based on the query using CLIP-based frame selection.
    
    Args:
        query: The text query to search for in the video
        video_path: Path to the video file
        start: Start time in the video
        end: End time in the video
        frame_retriever: An instance of CLIPKeyFrameExtractor
        
    Returns:
        List of retrieved frames in the format needed by the model
    """
    print(f"Retrieving frames for query: {query}")
    
    # Extract key frames using the provided frame retriever
    key_frames_data = frame_retriever.extract_key_frames(
        video_path=video_path,
        question=query,
        sample_rate=5,  # Sample every 5th frame to speed up processing
        top_k=2,        # Return 3 most relevant frames
        min_distance=10, # Minimum distance between frames
        start=start,
        end=end,
        visualize=True  # Don't visualize results
    )
    
    # Convert numpy array frames to PIL Images
    retrieved_frames = [Image.fromarray(frame) for frame in key_frames_data["frames"]]
    
    return retrieved_frames