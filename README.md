# Transform KH Bot - Speech To Speech Bot 

Transform KH Bot is a speech-to-speech and multi-modal Telegram bot that transcribes English or Khmer voice messages and text messages, translates them to English or Khmer, and generates English or Khmer speech. It leverages AI models for speech recognition, translation, and text-to-speech synthesis. Built on top of [TranscribeKH Bot](https://github.com/lyfeyvutha/transcribe-kh-bot).

## Features

- English voice message transcription using Whisper
- Khmer voice message transcription using CADT KhmerASR API
- English to Khmer & Khmer to Englishtranslation using TranslateKH API
- English and Khmer text-to-speech synthesis using MMS-TTS
- English & Khmer text inputs
- User-friendly Telegram interface

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lyfeyvutha/transform-kh-bot.git
   cd transform-kh-bot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following structure:
   ```bash
   {
      TELEGRAM_API_KEY="YOUR_TELEGRAM_BOT_TOKEN"
      VOICE_MESSAGE_FILE_PATH="path/to/save/voice/messages",
      TRANSLATE_KH_USERNAME="YOUR_TRANSLATE_KH_USERNAME",
      TRANSLATE_KH_PASSWORD="YOUR_TRANSLATE_KH_PASSWORD"
   }
   ```

## Usage

1. Start the bot:
   ```bash
   python main.py
   ```

2. Open Telegram and start a conversation with your bot.

3. Send a voice message to the bot.

4. The bot will process your message and reply with:
   - The original English transcription
   - The Khmer translation
   - A voice message with the Khmer speech

## Commands

- `/start`: Initialize the bot and get a welcome message
- `/help`: Display help information

## Technical Details

- Speech Recognition: OpenAI's Whisper (medium model)
- Translation: TranslateKH API
- Text-to-Speech: Facebook's MMS-TTS for Khmer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lyfeyvutha/transform-kh-bot/blob/main/LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper model
- CADT for the KhmerASR model
- Facebook for the MMS-TTS model
- TranslateKH for their translation API
- The Python-Telegram-Bot team for their excellent library

## Contact

For any queries or support, please open an issue in the GitHub repository.

## Author

[Chealyfey Vutha](https://github.com/lyfeyvutha)
