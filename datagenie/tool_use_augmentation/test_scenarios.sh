#!/bin/bash
# test_scenarios.sh - Run datagen for all scenarios and validate outputs
#
# Usage: ./datagenie/tool_use/test_scenarios.sh [samples_per_scenario]

set -e

SAMPLES=${1:-2}
OUTPUT_DIR="outputs/tool_use/test_run_$(date +%s)"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Tool-Use Scenario End-to-End Test"
echo "Samples per scenario: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

SCENARIOS=("single" "multistep" "multiturn" "relevance")

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "--- Running $scenario scenario ---"
    
    python datagenie/tool_use/datagen_tool_use.py \
        --scenario "$scenario" \
        --limit "$SAMPLES" \
        --batch_size 4 \
        2>&1 | tee "$OUTPUT_DIR/${scenario}_log.txt"
    
    # Move outputs to test directory
    mv outputs/tool_use/tool_use_sharegpt_${scenario}_*.jsonl "$OUTPUT_DIR/" 2>/dev/null || true
    mv outputs/tool_use/tool_use_results_${scenario}_*.jsonl "$OUTPUT_DIR/" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "Running Validation..."
echo "=============================================="

# Validate all outputs
for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "--- Validating $scenario ---"
    sharegpt_file=$(ls "$OUTPUT_DIR"/tool_use_sharegpt_${scenario}_*.jsonl 2>/dev/null | head -1)
    results_file=$(ls "$OUTPUT_DIR"/tool_use_results_${scenario}_*.jsonl 2>/dev/null | head -1)
    
    if [ -f "$sharegpt_file" ]; then
        valid_count=$(wc -l < "$sharegpt_file" | tr -d ' ')
        echo "  ShareGPT valid: $valid_count conversations"
    else
        echo "  WARNING: No ShareGPT output found"
    fi
    
    if [ -f "$results_file" ]; then
        total=$(wc -l < "$results_file" | tr -d '[:space:]')
        valid=$(grep -c '"valid": true' "$results_file" 2>/dev/null | tr -d '[:space:]' || echo "0")
        invalid=$(grep -c '"valid": false' "$results_file" 2>/dev/null | tr -d '[:space:]' || echo "0")
        echo "  Results: $total total, $valid valid, $invalid invalid"
        
        # Show validation reasons for failures
        if [ "$invalid" != "0" ] && [ -n "$invalid" ]; then
            echo "  Failure reasons:"
            grep '"valid": false' "$results_file" | jq -r '.validation_reason' 2>/dev/null | sort | uniq -c | head -5
        fi
    fi
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="

echo "Output files:"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No output files"

echo ""
echo "Run Python evaluation for detailed stats:"
echo "  python datagenie/tool_use/test_scenarios.py --eval $OUTPUT_DIR/tool_use_sharegpt_*.jsonl"
