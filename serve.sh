#!/bin/bash

PORT=4130

# Kill any existing process on the port
PID=$(lsof -t -i :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Killing existing process on port $PORT (PID: $PID)"
    kill $PID 2>/dev/null
    sleep 1
fi

echo "Starting server on http://localhost:$PORT/"
nohup uv run python -m http.server $PORT > /dev/null 2>&1 &
echo "Server running in background (PID: $!)"
