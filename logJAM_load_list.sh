#!/usr/bin/env bash
###############################################################################
# run_for_rxids.sh
#
# Examples
#   ./run_for_rxids.sh                       # default list, all commands
#   ./run_for_rxids.sh -s 2025-03-01         # default list, custom date, all cmds
#   ./run_for_rxids.sh -i -s 2025-03-01      # ingest only
#   ./run_for_rxids.sh -iv RX1 RX2           # ingest + varnalyzer on custom RXIDs
###############################################################################
set -euo pipefail

###############################################################################
# 0. Defaults
###############################################################################
DEFAULT_START="2025-04-01"
DEFAULT_RXIDS=(
  R1886469638 R1969419103 R1911703860 R1911703992 R1911704038 R1911703088
  R1886354843 R1911705114 R1946201411 R1911705088 R1886471404 R1956349185
  R1963244619 R1946890461 R1954124733 R1946203661 R1911088562 R1912226744
)

###############################################################################
# 1. Parse CLI options
###############################################################################
START_DATE="$DEFAULT_START"

# Command‑run flags – default “run everything”
RUN_INGEST=true
RUN_VARNALYZER=true
RUN_ANOMALY=true     # both LSTM & autoencoder

explicit_flag_set=false   # tracks if user specified -i/-v/-a

rxids=()   # will fill with positional args

while (( $# )); do
  case "$1" in
    -s|--start)
        [[ $# -lt 2 ]] && { echo "Missing date after $1"; exit 1; }
        START_DATE="$2"; shift 2;;
    -i) RUN_INGEST=true     RUN_VARNALYZER=false RUN_ANOMALY=false explicit_flag_set=true; shift;;
    -v) RUN_INGEST=false    RUN_VARNALYZER=true  RUN_ANOMALY=false explicit_flag_set=true; shift;;
    -a) RUN_INGEST=false    RUN_VARNALYZER=false RUN_ANOMALY=true  explicit_flag_set=true; shift;;
    -iv|-vi) RUN_INGEST=true RUN_VARNALYZER=true RUN_ANOMALY=false explicit_flag_set=true; shift;;
    -ia|-ai) RUN_INGEST=true RUN_VARNALYZER=false RUN_ANOMALY=true explicit_flag_set=true; shift;;
    -va|-av) RUN_INGEST=false RUN_VARNALYZER=true RUN_ANOMALY=true explicit_flag_set=true; shift;;
    -iva|-iav|-vai|-via|-aiv|-avi)
        RUN_INGEST=true RUN_VARNALYZER=true RUN_ANOMALY=true explicit_flag_set=true; shift;;
    --) shift; break ;;                 # end‑of‑options marker
    -h|--help)
        echo "Usage: $0 [-i|-v|-a|-iv|-va|-ia|-iva] [-s YYYY-MM-DD] [RXID ...]"
        exit 0 ;;
    *)  rxids+=("$1"); shift ;;
  esac
done

# If no RXIDs supplied, use defaults
(( ${#rxids[@]} )) || rxids=("${DEFAULT_RXIDS[@]}")

###############################################################################
# 2. One‑RXID pipeline
###############################################################################
process_one() {
  local rx="$1"
  echo "[$(date)] ==== START  $rx  (start=$START_DATE) ===="

  if $RUN_INGEST; then
    echo "[$(date)]  • ingest"
    python ingestion/log_ingest.py \
           --start_date "$START_DATE" -w 10 -d "/ccshare/logs/smplogs/$rx"
  fi

  if $RUN_VARNALYZER; then
    echo "[$(date)]  • varnalyzer"
    python analysis/varnalyzer.py \
           --start_time "$START_DATE" --table_name "$rx"
  fi

  if $RUN_ANOMALY; then
    echo "[$(date)]  • LSTM anomaly"
    python analysis/anomaly_detection/lstm_anomaly.py \
           --mode analyze --table_name "$rx"

    echo "[$(date)]  • autoencoder"
    python analysis/anomaly_detection/autoencoder.py \
           --mode analyze --table_name "$rx"
  fi

  echo "[$(date)] ==== DONE   $rx ===="
}

###############################################################################
# 3. Run up to 5 pipelines in parallel
###############################################################################
max_parallel=5
running=0

for rx in "${rxids[@]}"; do
  process_one "$rx" &   # background
  (( ++running >= max_parallel )) && { wait -n; (( running-- )); }
done

wait   # wait for remaining jobs
echo "All RXIDs processed."
