#!/bin/bash

# Ensure the environment variable is set
if [ -z "$LITELLM_API_KEY" ]; then
  echo "LITELLM_API_KEY is not set. Please set it before running this script."
  exit 1
fi

# Run 20 batches in parallel
for i in {0..39}
do
  echo "Running batch $i"
  python ./gen_sftdata.py $i > ./logs/log_batch_${i}.txt 2>&1 &
done

# Wait for all background processes to complete
wait
echo "All batches completed."
