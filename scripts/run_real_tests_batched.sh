#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Batched real-model test runner for M1 MacBook Air (8GB RAM).
#
# Runs the full pytest suite in isolated batches of $BATCH_SIZE tests.
# Each batch is a separate Python subprocess so macOS can reclaim MLX Metal
# memory between batches (prevents OOM from unified-memory fragmentation).
#
# Usage:
#   bash scripts/run_real_tests_batched.sh            # default batch size 50
#   BATCH_SIZE=25 bash scripts/run_real_tests_batched.sh   # smaller batches
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

BATCH_SIZE="${BATCH_SIZE:-50}"
export OPENHYDRA_USE_REAL_MODEL=1

cd "$(dirname "$0")/.." || exit 1

echo "════════════════════════════════════════════════════════════"
echo "  OpenHydra Batched Real-Model Test Runner"
echo "  BATCH_SIZE=$BATCH_SIZE  OPENHYDRA_USE_REAL_MODEL=1"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Collect all test node IDs ─────────────────────────────────────────
echo "Collecting tests..."
ALL_TESTS=$(python3 -m pytest --collect-only 2>/dev/null | grep "::")
TOTAL=$(echo "$ALL_TESTS" | wc -l | tr -d ' ')
NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "  Found $TOTAL tests → $NUM_BATCHES batches of up to $BATCH_SIZE"
echo ""

# ── Step 2: Run each batch as an isolated subprocess ──────────────────────────
BATCH_NUM=0
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
TOTAL_ERRORS=0
FAILED_BATCHES=""
OVERALL_START=$(date +%s)

# Write all tests to a temp file, then split
ALLTMP=$(mktemp)
echo "$ALL_TESTS" > "$ALLTMP"

while true; do
    BATCH_NUM=$((BATCH_NUM + 1))
    OFFSET=$(( (BATCH_NUM - 1) * BATCH_SIZE + 1 ))

    # Extract this batch's tests
    BATCH=$(sed -n "${OFFSET},$(( OFFSET + BATCH_SIZE - 1 ))p" "$ALLTMP")
    if [ -z "$BATCH" ]; then
        BATCH_NUM=$((BATCH_NUM - 1))
        break
    fi

    BATCH_COUNT=$(echo "$BATCH" | wc -l | tr -d ' ')
    echo "═══ Batch $BATCH_NUM / $NUM_BATCHES ($BATCH_COUNT tests) ═══"

    # Write batch test IDs to temp file
    BATCHTMP=$(mktemp)
    echo "$BATCH" > "$BATCHTMP"

    BATCH_START=$(date +%s)

    # Run pytest for this batch; capture output and exit code
    # -s to show print() output (Bangalore generation text)
    # --tb=short for readable tracebacks
    BATCH_OUTPUT=$(python3 -m pytest $(cat "$BATCHTMP" | tr '\n' ' ') \
        -v --tb=short -s 2>&1) || true
    BATCH_EXIT=$?

    BATCH_END=$(date +%s)
    BATCH_ELAPSED=$((BATCH_END - BATCH_START))

    # Parse the pytest summary line (e.g. "5 passed, 2 skipped, 1 failed")
    SUMMARY=$(echo "$BATCH_OUTPUT" | grep -E "^[=].*[=]$" | tail -1)
    echo "$BATCH_OUTPUT" | tail -20

    # Extract counts from summary
    B_PASSED=$(echo "$SUMMARY" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
    B_FAILED=$(echo "$SUMMARY" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
    B_SKIPPED=$(echo "$SUMMARY" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)
    B_ERRORS=$(echo "$SUMMARY" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo 0)

    # Default to 0 if empty
    B_PASSED=${B_PASSED:-0}; B_FAILED=${B_FAILED:-0}
    B_SKIPPED=${B_SKIPPED:-0}; B_ERRORS=${B_ERRORS:-0}

    TOTAL_PASSED=$((TOTAL_PASSED + B_PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + B_FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + B_SKIPPED))
    TOTAL_ERRORS=$((TOTAL_ERRORS + B_ERRORS))

    if [ "$B_FAILED" -gt 0 ] || [ "$B_ERRORS" -gt 0 ]; then
        FAILED_BATCHES="$FAILED_BATCHES $BATCH_NUM"
    fi

    echo "  → Batch $BATCH_NUM: ${B_PASSED} passed, ${B_FAILED} failed, ${B_SKIPPED} skipped (${BATCH_ELAPSED}s)"
    echo ""

    rm -f "$BATCHTMP"
done

rm -f "$ALLTMP"

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

# ── Step 3: Print aggregated summary ─────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  AGGREGATED RESULTS"
echo "════════════════════════════════════════════════════════════"
echo "  Batches:  $BATCH_NUM (batch size $BATCH_SIZE)"
echo "  Passed:   $TOTAL_PASSED"
echo "  Failed:   $TOTAL_FAILED"
echo "  Skipped:  $TOTAL_SKIPPED"
echo "  Errors:   $TOTAL_ERRORS"
echo "  Time:     ${OVERALL_ELAPSED}s ($(( OVERALL_ELAPSED / 60 ))m $(( OVERALL_ELAPSED % 60 ))s)"
if [ -n "$FAILED_BATCHES" ]; then
    echo "  FAILED BATCHES:$FAILED_BATCHES"
fi
echo "════════════════════════════════════════════════════════════"

# Exit non-zero if any failures
if [ "$TOTAL_FAILED" -gt 0 ] || [ "$TOTAL_ERRORS" -gt 0 ]; then
    exit 1
fi
exit 0
