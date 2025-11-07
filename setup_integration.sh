#!/bin/bash
# setup_integration.sh

echo "🚀 Setting up NBA-DFS-Tools Integration"

# Clone NBA-DFS-Tools if not exists
if [ ! -d "NBA-DFS-Tools" ]; then
    echo "📥 Cloning NBA-DFS-Tools repository..."
    git clone https://github.com/chanzer0/NBA-DFS-Tools.git
else
    echo "✅ NBA-DFS-Tools already exists"
fi

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Check if NBA-DFS-Tools has its own requirements
if [ -f "NBA-DFS-Tools/requirements.txt" ]; then
    pip install -r NBA-DFS-Tools/requirements.txt
    pip install nba_api pulp pandas
fi

echo "✅ Setup complete!"
echo "📝 Don't forget to update the path in config.py"