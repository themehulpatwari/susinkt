# Susinkt

An AI-powered YouTube Shorts recommendation system that learns your preferences and auto-(curates/scrolls) your feed.

## What It Does

Susinkt watches YouTube Shorts with you and learns what you like:
- **Predicts** which videos you'll enjoy using AI
- **Auto-scrolls** through videos (stays on good ones, skips boring ones)
- **Learns** from your viewing behavior to get better over time
- **Gesture control** - navigate with hand movements or head nods

## How It Works

1. Tell the AI what you want to watch (e.g., "funny cat videos" or "cooking tutorials")
2. The system analyzes each video and predicts if you'll like it
3. Automatically stays on good videos or scrolls past bad ones
4. Learns from which videos you actually watch
5. Gets smarter over time

## Features

- 🤖 AI predictions for each video
- 👋 Hand gesture controls (swipe to navigate)
- 👤 Head pose detection (nod to scroll)
- ⚡ Auto-scroll based on predictions
- 📈 Self-improving recommendations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
```

3. Run Chrome with remote debugging:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

4. Start the app:
```bash
python src/integration/main.py
```

## Requirements

- Python 3.8+
- Webcam
- Google Chrome
- YouTube Data API key
- Gemini API key

## License

See LICENSE file for details.
