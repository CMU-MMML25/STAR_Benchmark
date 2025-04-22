import argparse
import pickle
import pandas as pd
import numpy as np
import json
import os
import random


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def subsample_dataset(input_path, output_pkl, output_json, total_samples, seed=2025):  # Don't change the seed
    
    set_seed(seed)
    
    # Load the dataset
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add a 'category' field derived from question_id
    df['category'] = df['question_id'].apply(lambda x: x.split('_')[0])

    # Get normalized category distribution
    category_counts = df['category'].value_counts(normalize=True)
    print("Original category distribution:")
    print(category_counts)

    # Compute samples per category
    samples_per_category = {cat: int(prop * total_samples) for cat, prop in category_counts.items()}

    # Adjust rounding errors
    remaining = total_samples - sum(samples_per_category.values())
    if remaining > 0:
        for cat in category_counts.index[:remaining]:
            samples_per_category[cat] += 1

    print("\nSamples per category:")
    for cat, count in samples_per_category.items():
        print(f"{cat}: {count}")

    # Sample data
    sampled_indices = []
    for cat, count in samples_per_category.items():
        cat_indices = df[df['category'] == cat].index.tolist()
        sampled_cat_indices = np.random.choice(cat_indices, size=count, replace=False)
        sampled_indices.extend(sampled_cat_indices)

    # Extract sampled data
    sampled_df = df.loc[sampled_indices].copy()
    sampled_df = sampled_df.drop(columns=['category'])

    # Convert back to original list-of-dict format
    sampled_data = sampled_df.to_dict('records')

    # Save outputs
    pkl_path = output_pkl
    json_path = output_json

    with open(pkl_path, 'wb') as f:
        pickle.dump(sampled_data, f)

    with open(json_path, 'w') as f:
        json.dump(sampled_data, f, indent=2)

    print(f"\n✅ Saved {len(sampled_data)} examples to:")
    print(f"  • Pickle: {pkl_path}")
    print(f"  • JSON  : {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample STAR dataset with category balance.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output path for the .pkl file")
    parser.add_argument("--output_json", type=str, required=True, help="Output path for the .json file")
    parser.add_argument("--total_examples", type=int, required=True, help="Total number of examples to sample")

    args = parser.parse_args()

    subsample_dataset(args.input, args.output_pkl, args.output_json, args.total_examples)

"""

python subsample_STAR.py \
  --input /data/user_data/jamesdin/STAR/data/STAR_val.pkl \
  --output_pkl /data/user_data/jamesdin/STAR/data/STAR_val_1k.pkl \
  --output_json /data/user_data/jamesdin/STAR/data/STAR_val_1k.json \
  --total_examples 1000

python subsample_STAR.py \
  --input /data/user_data/jamesdin/STAR/data/STAR_test.pkl \
  --output_pkl /data/user_data/jamesdin/STAR/data/STAR_test_1k.pkl \
  --output_json /data/user_data/jamesdin/STAR/data/STAR_test_1k.json \
  --total_examples 1000


"""