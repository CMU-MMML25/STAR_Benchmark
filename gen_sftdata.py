import json
import os
import sys
from tqdm import tqdm
from openai import OpenAI

batch_idx = int(sys.argv[1])

# Initialize OpenAI client with custom base_url and API key
client = OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

# print("Loading data...")
# Load data from a JSON file
with open('videoespresso_train_video.json', 'r') as f:
    data_list = json.load(f)
# print(f"Loaded {len(data_list)} entries.")

data_list = data_list[5*(batch_idx):5*(batch_idx+1)]  # Limit to first 10 entries for testing

# Template prompt
base_prompt = """
You are tasked with transforming a raw video question answering (QA) dataset into a reasoning trace for training a video agent. You will be given a single data sample from the VideoEspresso dataset, which includes:
- A natural language question
- A natural language answer 
- A list of keyframes (file paths) 
- An evidence field describing what was visually observed at each frame, including objects and bounding boxes 

Your goal is to convert this into a reasoning trace that teaches a language model to:
- Think aloud about what information it needs to answer the question
- Formulate a visual question to retrieve that information (without knowing frame numbers in advance)
- Use "<look>" to wrap this visual query
- Receive the retrieved evidence as a list of bounding boxes and object names 
- Update its understanding based on the observation 
- Conclude with the answer in the form Answer: ...

Rules:
Do not assume knowledge of frame numbers. The model should discover relevant frames through natural questions (e.g., "What object is the Rabbid holding?", not "What happened in frame 2?"). 
Each reasoning step should follow this format: Step N: I need to know [specific information] <look> [natural language question to retrieve visual info] </look> <box_start> [bounding boxes] <box_end> [A brief explanation of how this changes the model's understanding]
Do not invent visual content not explicitly described in the evidence. The output should end with the full original answer: <answer> ... </answer>
Each <look> should be followed by a </look> tag, and each bounding box should be wrapped in <box_start> and <box_end> tags. Each <answer> should be wrapped in <answer> and </answer> tags.
 
Generate the output in this format:
{
  "output": "your reasoning trace with steps, <look> queries, evidence, and <answer>"
}

Now process this example:
"""

# Store results
results = []

# Loop through data and generate responses
for i, entry in enumerate(tqdm(data_list, file=sys.stdout)):
    # Prepare the prompt
    input_str = json.dumps(entry, ensure_ascii=False)
    prompt = base_prompt + input_str

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        # max_tokens=10000
    )

    # Extract and parse the result
    try:
        entry["reasoning"] = response.choices[0].message.content
        results.append(entry)
    except Exception as e:
        print(f"Error processing entry index: {i}, error: {e}", flush=True)
        print(f"Response: {response}", flush=True)
        print(f"Response content: {response.choices[0].message.content}", flush=True)
        print(f"Problematic entry: {entry['question']}", flush=True)
        continue

# Save results to a JSON file
with open(f'./output_data/batch_{batch_idx:03d}.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
