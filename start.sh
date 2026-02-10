#!/bin/bash

# Quick Start Script for AI Customer Intelligence Platform
# This script automates the setup and launch process

echo "🤖 AI Customer Intelligence Platform - Quick Start"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create .env if not exists
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  Please edit .env and add your OpenRouter API key!${NC}"
    echo -e "${YELLOW}Get your key at: https://openrouter.ai/${NC}"
    read -p "Press Enter after adding your API key..."
fi

# Train models
echo -e "\n${YELLOW}Training ML models...${NC}"
echo "This may take a few minutes..."
python backend.py --train

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
else
    echo -e "${RED}✗ Training failed. Check error messages above.${NC}"
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${RED}✗ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Port $port is available${NC}"
        return 0
    fi
}

# Check ports
echo -e "\n${YELLOW}Checking ports...${NC}"
check_port 8000
api_port_ok=$?

check_port 8501
dashboard_port_ok=$?

if [ $api_port_ok -ne 0 ] || [ $dashboard_port_ok -ne 0 ]; then
    echo -e "${RED}Please free up the required ports and try again${NC}"
    exit 1
fi

# Start services
echo -e "\n${GREEN}Starting services...${NC}"
echo "=================================================="

# Start API in background
echo -e "${YELLOW}Starting API server on port 8000...${NC}"
python backend.py --serve > api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API to start
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ API server is running${NC}"
else
    echo -e "${RED}✗ API server failed to start. Check api.log${NC}"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start Dashboard
echo -e "\n${YELLOW}Starting Streamlit dashboard on port 8501...${NC}"
streamlit run dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID"

# Wait for dashboard to start
sleep 3

echo -e "\n${GREEN}=================================================="
echo "✅ AI Customer Intelligence Platform is running!"
echo "==================================================${NC}"
echo ""
echo "📊 Dashboard: http://localhost:8501"
echo "🔌 API Docs:  http://localhost:8000/docs"
echo "❤️  Health:   http://localhost:8000/health"
echo ""
echo "Process IDs:"
echo "  API:       $API_PID"
echo "  Dashboard: $DASHBOARD_PID"
echo ""
echo "Logs:"
echo "  API:       tail -f api.log"
echo "  Dashboard: tail -f dashboard.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    kill $API_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Keep script running
wait
