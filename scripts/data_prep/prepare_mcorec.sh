#!/bin/bash

MCOREC_DATA_ROOT=$1

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
REPO_ROOT_PATH="$SCRIPT_PATH/../../"
NUM_WORKERS=4

for part in "train" "dev"; do
    FILLED_ROOT="$MCOREC_DATA_ROOT/data/mcorec_filled/$part"
    
    # FIll-in the gaps and concat all the tracks.
    python $SCRIPT_PATH/fill_mcorec_tracks.py \
        --session_dir "$MCOREC_DATA_ROOT/dev/*" \
        --output_root $FILLED_ROOT

    python $SCRIPT_PATH/create_mcorec_lhotse_manifests.py \
        --orig-root "$MCOREC_DATA_ROOT/$part" \
        --filled-root "$FILLED_ROOT/dev" \
        --output-cuts "$REPO_ROOT_PATH/manifests/mcorec_dev.jsonl.gz" \
        --num-workers 4
done
