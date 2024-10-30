#!/usr/bin/env bash
#   Use this script to test if a given TCP host/port are available

set -e

TIMEOUT=15
QUIET=0
HOST=""
PORT=""
CMD=()

print_help() {
  echo "
Usage:
  $0 host:port [--command [args...]]
  $0 [--help|--timeout=15|--quiet|--strict|--child|--host=HOST|--port=PORT|-- command args]
"
}

while [[ $# -gt 0 ]]
do
  case "$1" in
    --help)
    print_help
    exit 0
    ;;
    --timeout=*)
    TIMEOUT="${1#*=}"
    shift 1
    ;;
    --quiet)
    QUIET=1
    shift 1
    ;;
    --host=*)
    HOST="${1#*=}"
    shift 1
    ;;
    --port=*)
    PORT="${1#*=}"
    shift 1
    ;;
    --)
    shift
    CMD=("$@")
    break
    ;;
    *)
    HOST_PORT=(${1//:/ })
    HOST=${HOST_PORT[0]}
    PORT=${HOST_PORT[1]}
    shift 1
    ;;
  esac
done

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "Error: you need to provide a host and port to test."
  print_help
  exit 1
fi

for ((i=1;i<=TIMEOUT;i++)); do
  nc -z "$HOST" "$PORT" >/dev/null 2>&1 && break
  if [[ $i -eq TIMEOUT ]]; then
    echo "Timeout occurred after waiting $TIMEOUT seconds for $HOST:$PORT"
    exit 1
  fi
  sleep 1
done

if [[ $QUIET -ne 1 ]]; then
  echo "$HOST:$PORT is available after $i seconds"
fi

if [[ ${#CMD[@]} -gt 0 ]]; then
  exec "${CMD[@]}"
fi

exit 0
