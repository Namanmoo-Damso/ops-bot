# ops-bot

Bot client that joins LiveKit rooms to simulate user conversations with the agent.

## Setup

1. Copy the environment template:
   ```bash
   cp bot.env.example .env
   ```

2. Fill in your credentials in `.env`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Bot

```bash
cd bot
python stress_test_bots.py -n 1
```

The bot will:
1. Create a bot session via the ops-api
2. Join the LiveKit room as a participant
3. Send an initial greeting in Korean
4. Listen to the agent's speech and respond with predefined phrases
