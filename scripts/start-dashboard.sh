#!/bin/bash
# Start the Apart Dashboard (both backend and frontend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting Apart Dashboard..."
echo "Project root: $PROJECT_ROOT"

# Check if we're in the right place
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Run from project root."
    exit 1
fi

# Install npm dependencies if needed
if [ ! -d "$PROJECT_ROOT/dashboard/node_modules" ]; then
    echo "Installing dashboard dependencies..."
    cd "$PROJECT_ROOT/dashboard"
    npm install
fi

# Start backend server in background
echo "Starting backend server on http://localhost:8000..."
cd "$PROJECT_ROOT/src"
uv run python -m server.app &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend dev server
echo "Starting frontend on http://localhost:3000..."
cd "$PROJECT_ROOT/dashboard"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Dashboard is starting..."
echo "  Backend API: http://localhost:8000"
echo "  Frontend UI: http://localhost:3000"
echo "  API Docs:    http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C and kill both processes
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait
