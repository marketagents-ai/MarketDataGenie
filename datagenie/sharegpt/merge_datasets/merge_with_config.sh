#!/bin/bash
# Simple script to merge datasets using the configuration file

CONFIG_FILE="merge_config.yml"

# Read target repo (remove quotes and trim)
TARGET_REPO=$(grep "target_repo:" $CONFIG_FILE | cut -d':' -f2 | tr -d '"' | tr -d "'" | xargs)

# Read system message (handle multi-line and quotes properly)
SYSTEM_MSG=$(grep "system_message:" $CONFIG_FILE | cut -d':' -f2- | sed 's/^[[:space:]]*//' | tr -d '"' | tr -d "'")

echo "Merging datasets to: $TARGET_REPO"
echo "System message: $SYSTEM_MSG"

# Build command from config
CMD="python merge_datasets.py --target-repo $TARGET_REPO --system-message \"$SYSTEM_MSG\""

# Add hub datasets - read all lines between hub_datasets: and the next top-level key
echo "Reading hub datasets..."
hub_datasets_section=$(sed -n '/^hub_datasets:/,/^[^[:space:]]/p' $CONFIG_FILE | grep -v "^hub_datasets:" | grep -v "^[^[:space:]]")

while IFS= read -r line; do
    if [[ $line =~ ^[[:space:]]*-[[:space:]]*name:[[:space:]]*\"([^\"]+)\" ]]; then
        REPO_NAME="${BASH_REMATCH[1]}"
        echo "  Found dataset: $REPO_NAME"
        SPLIT=$(grep -A1 "name: \"$REPO_NAME\"" $CONFIG_FILE | grep "split:" | cut -d':' -f2 | tr -d ' ' | tr -d '"' | tr -d "'")
        CMD="$CMD --add-hub \"$REPO_NAME\" \"$SPLIT\""
    fi
done <<< "$hub_datasets_section"

# Add local files (only if they exist and are not empty)
echo "Reading local files..."
local_files_section=$(sed -n '/^local_files:/,/^[^[:space:]]/p' $CONFIG_FILE | grep -v "^local_files:" | grep -v "^[^[:space:]]")

while IFS= read -r line; do
    if [[ $line =~ ^[[:space:]]*-[[:space:]]*path:[[:space:]]*\"([^\"]+)\" ]]; then
        FILE_PATH="${BASH_REMATCH[1]}"
        SOURCE=$(grep -A1 "path: \"$FILE_PATH\"" $CONFIG_FILE | grep "source:" | cut -d':' -f2 | tr -d ' ' | tr -d '"' | tr -d "'")
        # Only add if file exists
        if [ -f "$FILE_PATH" ]; then
            echo "  Found local file: $FILE_PATH"
            CMD="$CMD --add-file \"$FILE_PATH\" \"$SOURCE\""
        else
            echo "Warning: Local file not found: $FILE_PATH (skipping)"
        fi
    fi
done <<< "$local_files_section"

# Add upload flag if enabled
if grep -q "upload: true" $CONFIG_FILE; then
    CMD="$CMD --upload"
fi

echo "Executing: $CMD"
eval $CMD
