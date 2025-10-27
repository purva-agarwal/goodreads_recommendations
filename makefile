python -m venv venv
source venv/bin/activate(mac/linux) or         venv\Scripts\activate(windows)
pip install -r requirements.txt
deactivate
# inside the agentic-reel-bot directory
# Build and run
docker-compose up --build

# Stop
docker-compose down