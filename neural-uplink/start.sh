#!/bin/bash
# Neural Uplink — Baltimore Node Launcher
# Usage: bash start.sh [start|stop|restart|status|logs]

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$NODE_DIR"

ACTION="${1:-start}"

start() {
    echo "[Neural Uplink] Starting Baltimore Node..."
    mkdir -p logs

    echo "[*] Launching Neural Uplink Core (port 7000)..."
    nohup python3 core/neural_uplinkd.py > logs/neural_uplinkd.log 2>&1 &
    sleep 2

    echo "[*] Launching agents..."
    nohup python3 agents/dialog.py > logs/agent-dialog.log 2>&1 &
    sleep 1
    nohup python3 agents/data.py > logs/agent-data.log 2>&1 &
    sleep 1
    nohup python3 agents/ops.py > logs/agent-ops.log 2>&1 &
    sleep 1
    nohup python3 agents/worldbuilder.py > logs/agent-worldbuilder.log 2>&1 &
    sleep 1

    echo "[*] Launching servers..."
    nohup python3 servers/zo_gateway.py > logs/zo-gateway.log 2>&1 &
    sleep 1
    nohup python3 servers/zo_control.py > logs/zo-control.log 2>&1 &
    sleep 1

    echo "[*] Launching watchdog..."
    nohup python3 infra/watchdog.py > logs/watchdog.log 2>&1 &
    sleep 1

    echo ""
    echo "[+] Neural Uplink — Baltimore Node is LIVE"
    echo ""
    echo " Public API:  http://localhost:8000/gateway/status"
    echo " Control:     http://localhost:8001/system/health"
    echo " Uplink Core: http://localhost:7000/health"
    echo ""
}

stop() {
    echo "[Neural Uplink] Stopping Baltimore Node..."
    pkill -f "neural_uplinkd.py" 2>/dev/null
    pkill -f "agents/dialog.py" 2>/dev/null
    pkill -f "agents/data.py" 2>/dev/null
    pkill -f "agents/ops.py" 2>/dev/null
    pkill -f "agents/worldbuilder.py" 2>/dev/null
    pkill -f "zo_gateway.py" 2>/dev/null
    pkill -f "zo_control.py" 2>/dev/null
    pkill -f "watchdog.py" 2>/dev/null
    echo "[*] All processes stopped"
}

status() {
    echo "=== Baltimore Node Status ==="
    for port in 7000 7101 7102 7103 7104 8000 8001; do
        nc -z localhost $port 2>/dev/null && echo "  port $port: UP" || echo "  port $port: DOWN"
    done
    echo ""
}

logs() {
    SERVICE="${2:-all}"
    if [ "$SERVICE" = "all" ]; then
        echo "=== All Logs ==="
        for f in logs/*.log; do
            echo "--- $f ---"
            tail -20 "$f" 2>/dev/null || echo "(empty)"
        done
    else
        echo "--- logs/${SERVICE}.log ---"
        tail -30 "logs/${SERVICE}.log" 2>/dev/null
    fi
}

case $ACTION in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 2; start ;;
    status)  status ;;
    logs)    logs "$@" ;;
    *)
        echo "Usage: bash start.sh [start|stop|restart|status|logs <service>]"
        ;;
esac