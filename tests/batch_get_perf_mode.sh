#!/usr/bin/env bash
# Shared mode selection for the single-shape and sweep runners. Does not consume
# arguments: the Python benchmark parses the same flags for direct invocation.
kvaf_batch_get_parse_mode() {
  local explicit_mode=false raw_flag=false
  batch_get_mode="${KVAF_BATCH_GET_MODE:-}"
  if [[ -z "$batch_get_mode" ]]; then
    batch_get_mode=lookup
    if [[ "${KVAF_BATCH_GET_RAW_ADDRESSES:-0}" == 1 ]]; then
      batch_get_mode=raw
    fi
  fi
  while (($# > 0)); do
    case "$1" in
      --batch-get-mode)
        if (($# < 2)); then
          echo "ERROR: --batch-get-mode requires lookup, no-pipeline, or raw" >&2
          return 2
        fi
        explicit_mode=true
        batch_get_mode="$2"
        shift
        ;;
      --batch-get-mode=*)
        explicit_mode=true
        batch_get_mode="${1#*=}"
        ;;
      --raw-addresses)
        raw_flag=true
        batch_get_mode=raw
        ;;
    esac
    shift
  done
  if [[ "$explicit_mode" == true && "$raw_flag" == true ]]; then
    echo "ERROR: --batch-get-mode and --raw-addresses are mutually exclusive" >&2
    return 2
  fi
  case "$batch_get_mode" in
    lookup|no-pipeline) address_mode=lookup ;;
    raw) address_mode=raw ;;
    *)
      echo "ERROR: --batch-get-mode / KVAF_BATCH_GET_MODE must be lookup, no-pipeline, or raw: $batch_get_mode" >&2
      return 2
      ;;
  esac
}
