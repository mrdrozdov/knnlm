# Step 1. Save train data.

```
python eval_lm.py data-bin/wikitext-103 \
    --path wt103_checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap dstore_train/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 103225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore
```

This saves some values that were ignored in KNN-LM:
- key - The vector associated with source token.
- src - The source token. (new)
- tgt - The true next word. (this was val before)
- dist - The probability given to the true next word. (new)

This step takes about 6 hours with the following settings:

```
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45GB
```

# Step 2. Build KNN and index.

```
python build_dstore.py \
    --dstore_mmap dstore_train/dstore \
    --dstore_size 103225485 \
    --faiss_index dstore_train/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0
```

This step takes about 15 hours with following settings:

```
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45GB
```

# Step 3. Verify KNN and compute perplexity.

First, get relevant vectors from the validation data.

```
python eval_lm.py data-bin/wikitext-103 \
    --path wt103_checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --context-window 2560 --no-min-context \
    --dstore-mmap dstore_valid/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 217646 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore
```

Then run the eval in an "offline" way. These steps make it easier to run future eval without recomputing feature vectors.

Note: Random access with memmap is way too slow with large arrays, so always use `--knn-sim-func "do_not_recomp_l2"`. If you do read from memmap, then make sure to use `--knn-lookup-mode unique` which will remove some redundant reads and speed things up.

```
python offline_eval.py \
    --bsize 200 \
    --dstore dstore_valid \
    --dstore-size 217646 \
    --knn-lookup-mode unique \
    --knn-dstore dstore_train \
    --knn-dstore-size 103225485 \
    --knn-sim-func "do_not_recomp_l2" \
    --knn --k 1024
```

This should output the perplexity w/o KNN, and multiple perplexity values w/ KNN according to the interpolation used.

# Step 4. Start building the allrank data.

# Step 4a. Create new data splits from validation data.

```
python split_data.py \
    --dstore dstore_valid \
    --dstore-size 217646 \
    --output from_dstore_valid-20210214 \
    --tr-size 100000 \
    --va-size 100000 \
    --te-size 7646
```

# Step 4b. Cache vectors and write allrank data.

First, cache the neighbors for the training data.

```
python offline_eval.py \
    --bsize 1000 \
    --dstore from_dstore_valid-20210214/tr \
    --dstore-size 100000 \
    --knn-lookup-mode unique \
    --knn-dstore dstore_train \
    --knn-dstore-size 103225485 \
    --knn-sim-func "do_not_recomp_l2" \
    --knn --k 1024 \
    --save --save-to from_dstore_valid-20210214/lookup_tr

python offline_eval.py \
    --bsize 1000 \
    --dstore from_dstore_valid-20210214/va \
    --dstore-size 100000 \
    --knn-lookup-mode unique \
    --knn-dstore dstore_train \
    --knn-dstore-size 103225485 \
    --knn-sim-func "do_not_recomp_l2" \
    --knn --k 1024 \
    --save --save-to from_dstore_valid-20210214/lookup_va

python offline_eval.py \
    --bsize 1000 \
    --dstore from_dstore_test-20210214 \
    --dstore-size 245569 \
    --knn-lookup-mode unique \
    --knn-dstore dstore_train \
    --knn-dstore-size 103225485 \
    --knn-sim-func "do_not_recomp_l2" \
    --knn --k 1024 \
    --save --save-to from_dstore_test-20210214/lookup

# For some reason, only 217646 tokens were computed for test.
```

Then sample from here to create the new data.

```
python build_allrank_data.py \
    --output from_dstore_valid-20210214/allrank \
    --tr-dstore from_dstore_valid-20210214/tr \
    --tr-dstore-size 100000 \
    --tr-lookup from_dstore_valid-20210214/lookup_tr \
    --tr-lookup-k 1024 \
    --ntrain 30000 \
    --va-dstore from_dstore_valid-20210214/va \
    --va-dstore-size 100000 \
    --va-lookup from_dstore_valid-20210214/lookup_va \
    --va-lookup-k 1024 \
    --nvalid 10000 \
    --va-shuffle \
    --k 64

python build_allrank_data.py --test-only \
    --output from_dstore_test-20210214/allrank \
    --va-dstore from_dstore_test-20210214 \
    --va-dstore-size 245569 \
    --va-lookup from_dstore_test-20210214/lookup \
    --va-lookup-k 1024 \
    --nvalid -1 \
    --k 64
```

The file this creates is roughly 20GB.

# Step 5. Create the config for allrank.

There are some key values to modify.

## dstore

```
# The query ids are used to access the feature vectors here.

"dstore": {
  "path": "/abspath/to/dstore_train",
  "dstore_size": 103225485,
  "vec_size": 1024,
  "enabled": true
},

# TODO: Should have a separate entry for train and validation.
```

## other

```
"slate_length": 128 # This should match the value of k used.
```

Build the dataset from the validation data! Validation has 200k tokens. Use half for training, some for val, and some for test.

Do as much as you can on dexter. Transfer datasets to gypsum to run LTR experiments.