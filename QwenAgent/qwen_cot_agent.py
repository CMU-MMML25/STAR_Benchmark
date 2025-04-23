import argparse
import gc
import json
import os
import re
from collections import defaultdict
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from frame_retriever import CLIPKeyFrameExtractor
from qwen_vl_utils import process_vision_info
from video_qa_dataset import VideoQADataset

import prompts

class LookEndTokenStoppingCriteria(StoppingCriteria):
    """Stopping criteria that stops generation when </look> token is generated."""
    def __init__(self, tokenizer, input_length, stop_string="</look>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.input_length = input_length
        
    def __call__(self, input_ids, scores, **kwargs):
        # For each sequence in the batch
        for input_id in input_ids:
            # Only look at newly generated tokens
            if len(input_id) <= self.input_length:
                continue
                
            # Decode the generated text (keeping special tokens)
            generated_text = self.tokenizer.decode(
                input_id[self.input_length:], 
                skip_special_tokens=False
            )
            
            # Simple string match for the stop string
            if self.stop_string in generated_text:
                return True
        return False

class QwenCoTAgent:
    """Chain-of-Thought Agent using Qwen-VL model with frame retrieval capabilities."""
    
    def __init__(self, num_frames=8, prompt_template='user_prompt', clip_model='ViT', n_retrieved_frames_per_step=2, min_distance=10, similarity_threshold=None, device="cuda"):
        """
        Initialize the QwenCoTAgent.
        
        Args:
            num_frames: Number of frames to sample from the video initially
            device: Device to run the model on
        """
        self.device = device
        self.num_frames = num_frames
        
        # Initialize model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.model.eval()
        
        # reference: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
        available_clip_models = {
            "ViT": {"clip_model_name": "ViT-H-14-378-quickgelu", "pretrained": "laion2b_s34b_b79k"},
            "EVA-CLIP": {"clip_model_name": "EVA02-E-14-plus", "pretrained": "laion2b_s9b_b144k"},
            "SigLIP": {"clip_model_name": "ViT-SO400M-14-SigLIP-384", "pretrained": "webli"},
        }
        assert clip_model in available_clip_models, f"Unsupported CLIP model: {clip_model}, available models: {available_clip_models.keys()}"
        clip_model_name = available_clip_models[clip_model]["clip_model_name"]
        pretrained = available_clip_models[clip_model]["pretrained"]
        
        # Initialize frame retriever
        self.frame_retriever = CLIPKeyFrameExtractor(
            clip_model_name=clip_model_name,
            pretrained=pretrained,
            top_k=n_retrieved_frames_per_step,
            sample_rate=8,  # sample every 8th frame from the raw video
            similarity_threshold=similarity_threshold,
            min_distance=min_distance,
        )
        
        # Load prompts
        from prompts import system_prompt
        self.system_prompt = system_prompt
        try:
            self.user_prompt_template = getattr(prompts, prompt_template)
        except AttributeError:
            raise ValueError(f"Prompt template '{prompt_template}' not found in prompts.py")
    
    @torch.no_grad()
    def retrieve_frames(self, query: str, video_path: str, start: float, end: float, visualize: bool = False) -> List:
        """
        Retrieve relevant frames based on the query.
        
        Args:
            query: The text query to search for in the video
            video_path: Path to the video file
            start: Start time in the video
            end: End time in the video
            
        Returns:
            List of retrieved frames as PIL images
        """
        
        # Extract key frames using the provided frame retriever
        key_frames_data = self.frame_retriever.extract_key_frames(
            video_path=video_path,
            question=query,
            start=start,
            end=end,
            visualize=visualize  # Visualize results
        )
        
        # Convert numpy array frames to PIL Images
        from PIL import Image
        retrieved_frames = [Image.fromarray(frame) for frame in key_frames_data["frames"]]
        
        key_frames_data["frames"] = retrieved_frames
        
        return key_frames_data
    
    @torch.no_grad()
    def video_qa(self, video_path: str, question: str, choices: List[str], 
                 start: float, end: float, max_iterations: int = 5, verbose: bool = False) -> str:
        """
        Run the chain-of-thought video QA with look-retrieve functionality.
        
        Args:
            video_path: Path to the video file
            question: The question to answer
            choices: List of possible answers
            start: Start time in the video
            end: End time in the video
            max_iterations: Maximum number of look-retrieve cycles
            
        Returns:
            The model's final response
        """
        # Format the user prompt with the question and choices
        formatted_user_prompt = self.user_prompt_template.format(
            question=question,
            choices=''.join([f"{i+1}. {c}" + chr(10) for i, c in enumerate(choices)])
        )

        # Initial messages with video
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "video_start": start,
                        "video_end": end,
                        "nframes": self.num_frames
                    },
                    {"type": "text", "text": formatted_user_prompt},
                ],
            }
        ]

        # Prepare the initial template only once
        base_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        current_text = base_text  # Text to be used for this iteration
        
        # Initialize variables
        full_response = ""  # The accumulated assistant's response
        image_inputs = None   # Will hold retrieved frames
        total_retrieved_frames = 0
        retrieved_frames_info = []
        
        # Extract initial video inputs
        _, video_inputs = process_vision_info(messages)
        
        if verbose:
            print(f"[Start] Starting generation, max iterations: {max_iterations}")

        # Perform iterative generation with look-retrieve cycles
        for iteration in range(max_iterations):
            
            current_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Prepare inputs for the processor
            inputs = self.processor(
                text=[current_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Set up stopping criteria to find </look> tokens in newly generated content
            input_length = len(inputs.input_ids[0])
            look_stopping_criteria = LookEndTokenStoppingCriteria(
                self.processor.tokenizer,
                input_length=input_length,
                stop_string="</look>"
            )
            
            # Generate text with stopping at </look>
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                stopping_criteria=StoppingCriteriaList([look_stopping_criteria]),
            )

            # Decode the generated text
            input_length = len(inputs.input_ids[0])
            generated_ids = outputs[0]
            new_token_ids = generated_ids[input_length:]

            # Decode only the new tokens to text
            new_content = self.processor.tokenizer.decode(
                new_token_ids, 
                skip_special_tokens=False, 
                clean_up_tokenization_spaces=False
            )
            
            if verbose:
                print(f"[Iteration {iteration}] New content: {new_content}")

            # Check if we have a look query
            look_match = re.search(r"<look>(.*?)</look>", new_content)
            if look_match:
                # Extract the query
                query = look_match.group(1).strip()
                
                # Update full response and current text
                full_response += new_content
                current_text += new_content
                
                if verbose:
                    print(f"[Iteration {iteration}] Retrieved frames for query: {query}")
                
                # Retrieve frames based on the query
                key_frames_data = self.retrieve_frames(query, video_path, start, end, visualize=verbose)
                retrieved_frames = key_frames_data["frames"]
                
                retrieved_frames_info.append(
                    {
                        "timestamps": [float(t) for t in key_frames_data["timestamps"]],
                        "scores": [float(s) for s in key_frames_data["scores"]],
                        "indices": [int(i) for i in key_frames_data["indices"]],
                    }
                )  # Convert to json serializable format

                # Add the retrieved frames to image_inputs
                if not image_inputs:
                    image_inputs = retrieved_frames
                else:
                    image_inputs.extend(retrieved_frames)
                
                if messages[-1]['role'] != 'assistant':
                    messages.append({"role": "assistant", "content": []})
                
                messages[-1]['content'].append({"type": "text", "text": new_content})
                
                # Add vision markers for each new frame in the format Qwen expects
                for frame in retrieved_frames:
                    messages[-1]['content'].append({"type": "image", "image": "dummy_path"})
                    if verbose:
                        print(f"[Iteration {iteration}] Retrieved frame: {str(frame)}")
                    
                full_response += ''.join([f"<frame>{total_retrieved_frames+i}</frame>" for i in range(len(retrieved_frames))]) + '\n'
                total_retrieved_frames += len(retrieved_frames)
                
            else:
                # No more look queries, finalize the response
                full_response += new_content
                break

        return full_response, retrieved_frames_info

    def parse_answer(self, response: str) -> int:
        """
        Parse the answer index from the response.
        
        Args:
            response: The model's response
            
        Returns:
            The parsed answer index (1-4)
        """
        # Look for "Answer: <index>" pattern
        answer_match = re.search(r"Answer:\s*(\d+)", response)
        if answer_match:
            return int(answer_match.group(1))
        
        # Fallback: look for any number in the response
        number_match = re.search(r"\d+", response)
        if number_match:
            return int(number_match.group(0))
        
        # If no number found, default to 1
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenCoTAgent STAR Benchmark Inference")

    parser.add_argument("--val_pkl", type=str, default="/data/user_data/jamesdin/STAR/data/STAR_val.pkl", help="Path to validation .pkl file")
    parser.add_argument("--video_dir", type=str, default="/data/user_data/jamesdin/STAR/data/Charades_v1_480", help="Path to video directory")
    parser.add_argument("--results_file", type=str, default="analysis/qwen_cot_results.jsonl", help="Path to write model predictions")
    parser.add_argument("--final_accuracy_file", type=str, default="analysis/qwen_cot_final_accuracy.txt", help="Path to write final accuracy results")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of video frames to sample for inference")
    parser.add_argument("--prompt_template", type=str, default="user_prompt", help="Prompt for the model to generate reasoning chain")
    parser.add_argument("--clip_model", type=str, default="ViT", help="CLIP model type for frame retrieval")
    parser.add_argument("--n_retrieved_frames_per_step", type=int, default=2, help="Number of video frames to retrieve at each step")
    parser.add_argument("--min_distance", type=int, default=10, help="Minimum distance between retrieved frames")

    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset and dataloader
    dataset = VideoQADataset(args.val_pkl, video_dir=args.video_dir, sampling_fps=4, num_frames=args.num_frames, use_fps=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize the agent
    video_qa_agent = QwenCoTAgent(
        num_frames=args.num_frames,
        prompt_template=args.prompt_template,
        clip_model=args.clip_model,
        n_retrieved_frames_per_step=args.n_retrieved_frames_per_step,
        min_distance=args.min_distance,
        device=device,
    )

    # Initialize accuracy tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    # Open results file in append mode
    with open(args.results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for i, batch in enumerate(pbar):
                start_time = perf_counter()

                # Extract batch data
                start = batch["start"][0].item()
                end = batch["end"][0].item()
                question = batch["question"][0]
                choices = batch["choices"]
                choices = [c[0] for c in choices]
                answer_idx = batch["answer_idx"][0]
                category = batch["category"][0]
                question_id = batch["question_id"][0]
                video_path = batch["video_path"][0]

                # Get model prediction
                full_response, retrieved_frames_info = video_qa_agent.video_qa(
                    video_path, question, choices, start, end
                )

                # Parse the answer
                pred_answer = video_qa_agent.parse_answer(full_response)
                # Convert to 0-indexed for evaluation
                pred_index = pred_answer - 1

                end_time = perf_counter()

                # Save result to JSONL file
                json_record = {
                    "example_id": i,
                    "question_id": question_id,
                    "question": question,
                    "choices": choices,
                    "pred_ans_idx": pred_index,
                    "true_index": answer_idx.item(),
                    "category": category,
                    "raw_response": full_response,
                    "inference_time": (end_time - start_time),
                    "retrieved_frames_info": retrieved_frames_info,
                }

                results_f.write(json.dumps(json_record) + "\n")
                results_f.flush()

                # Update accuracy counters
                category_total[category] += 1
                if pred_index == answer_idx:
                    category_correct[category] += 1

                # Compute overall accuracy
                overall_acc = sum(category_correct.values()) / sum(category_total.values())

                # Update tqdm bar with accuracy
                accuracy_info = {cat: f"{category_correct[cat] / category_total[cat]:.3f}" for cat in category_total}
                accuracy_info["Overall"] = f"{overall_acc:.3f}"
                pbar.set_postfix(accuracy_info)

    # Save final category-wise accuracy to a text file
    with open(args.final_accuracy_file, "w") as acc_f:
        acc_f.write("Final Category-Wise Accuracy:\n")
        for category in category_total:
            acc_f.write(f"{category}: {category_correct[category] / category_total[category]:.4f}\n")
        acc_f.write(f"\nOverall accuracy: {sum(category_correct.values()) / sum(category_total.values()):.4f}\n")

    print(f"Results saved to {args.results_file} and final accuracy saved to {args.final_accuracy_file}")


"""

python QwenAgent/qwen_cot_agent.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val_1k.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/qwen_cot_agent_(user_prompt,n_retrieved=2,min_distance=20)_results_1k.jsonl" \
    --final_accuracy_file "analysis/qwen_cot_agent_(user_prompt,n_retrieved=2,min_distance=20)_final_accuracy_kl.txt" \
    --num_frames 8 \
    --prompt_template "user_prompt" \
    --n_retrieved_frames_per_step 2 \
    --min_distance 20

python QwenAgent/qwen_cot_agent.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val_1k.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/qwen_cot_agent_(harder_user_prompt,n_retrieved=1,min_distance=20)_results_1k.jsonl" \
    --final_accuracy_file "analysis/qwen_cot_agent_(harder_user_prompt,n_retrieved=1,min_distance=20)_final_accuracy_kl.txt" \
    --num_frames 8 \
    --prompt_template "harder_user_prompt" \
    --n_retrieved_frames_per_step 1 \
    --min_distance 20
    
python QwenAgent/qwen_cot_agent.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val_1k.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/qwen_cot_agent_(prompt3,n_retrieved=3,min_distance=10)_results_1k.jsonl" \
    --final_accuracy_file "analysis/qwen_cot_agent_(prompt3,n_retrieved=3,min_distance=10)_final_accuracy_kl.txt" \
    --num_frames 8 \
    --prompt_template "prompt3" \
    --n_retrieved_frames_per_step 3 \
    --min_distance 10

python QwenAgent/qwen_cot_agent.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val_1k.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/qwen_cot_agent_(SigLIP,prompt3,n_retrieved=3,min_distance=10)_results_1k.jsonl" \
    --final_accuracy_file "analysis/qwen_cot_agent_(SigLIP,prompt3,n_retrieved=3,min_distance=10)_final_accuracy_kl.txt" \
    --num_frames 8 \
    --prompt_template "prompt3" \
    --clip_model "SigLIP" \
    --n_retrieved_frames_per_step 3 \
    --min_distance 10

"""
