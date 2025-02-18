import json
import ssl
import certifi
import logging
import requests
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import VitsModel, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torch
import scipy.io.wavfile
import os
import librosa
import websockets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LANGUAGE_NAMES = {
    'en': 'English',
    'km': 'Khmer'
}

class TransformKHBot:
    def __init__(self):
        self.telegram_api_key = os.environ.get('TELEGRAM_API_KEY')
        self.voice_message_file_path = os.environ.get('VOICE_MESSAGE_FILE_PATH')
        self.translate_kh_username = os.environ.get('TRANSLATE_KH_USERNAME')
        self.translate_kh_password = os.environ.get('TRANSLATE_KH_PASSWORD')
        self.khmer_asr_websocket_url = os.environ.get('KHMER_ASR_WEBSOCKET_URL')

        if not all([self.telegram_api_key, self.voice_message_file_path, 
                    self.translate_kh_username, self.translate_kh_password,
                    self.khmer_asr_websocket_url]):
            raise ValueError("Missing required environment variables.")

        self.application = Application.builder().token(self.telegram_api_key).build()
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_khmer_tts_model()
        self.initialize_english_tts_model()
        self.initialize_whisper_model()

    def initialize_khmer_tts_model(self):
        logger.info("Initializing MMS model for Khmer TTS...")
        self.mms_model = VitsModel.from_pretrained("facebook/mms-tts-khm").to(self.device)
        self.khmer_tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
        logger.info("Khmer TTS model initialized successfully.")
    
    def initialize_english_tts_model(self):
        logger.info("Initializing MMS model for English TTS...")
        self.english_tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device)
        self.english_tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        logger.info("English TTS model initialized successfully.")

    def initialize_whisper_model(self):
        logger.info("Initializing Whisper model...")
        try:
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(self.device)
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            self.whisper_model.eval()
            logger.info("Whisper model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {e}", exc_info=True)
            raise

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
    
    def run(self):
        self.setup_handlers()
        logger.info("Bot is running...")
        self.application.run_polling()

    async def start_command(self, update: Update, context):
        await update.message.reply_text(
            "Welcome to Transform KH Bot\n"
            "Send a voice message, and the bot will convert it to text, translate to Khmer, and generate Khmer speech for you.\n"
            "Commands: \n"
            "/help - help information\n"
        )

    async def help_command(self, update: Update, context):
        await update.message.reply_text(
        "Send a voice message to get it transcribed and translated to Khmer.\n"
        "Or send a text message in Khmer to get it translated to English.")

    def preprocess_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = librosa.effects.trim(audio, top_db=20)[0]
        return audio

    async def transcribe_audio(self, audio_path):
        audio = self.preprocess_audio(audio_path)
        input_features = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        
        # Detect language and transcribe
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                input_features, 
                task="transcribe", 
                return_timestamps=False,
            )

        transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Check if the transcription is empty or contains non-English characters
        if not transcription or any(ord(char) > 127 for char in transcription):
            # Assume Khmer and use Khmer ASR
            async with websockets.connect(self.khmer_asr_websocket_url) as websocket:
                raw_audio, sr = librosa.load(audio_path, sr=16000)
                audio_bytes = (raw_audio * 32767).astype('int16').tobytes()
                await websocket.send(audio_bytes)
                result = await websocket.recv()
                result_json = json.loads(result)
                
                if 'partial' in result_json:
                    transcription = result_json['partial']
                elif 'text' in result_json:
                    transcription = result_json['text']
                else:
                    transcription = str(result_json)
            detected_language = 'km'
        else:
            detected_language = 'en'
        
        return transcription, detected_language

    async def handle_voice(self, update: Update, context):
        message = await update.message.reply_text("Processing your voice message")

        try:
            file = await update.message.voice.get_file()
            await file.download_to_drive(self.voice_message_file_path)
            
            transcription, detected_language = await self.transcribe_audio(self.voice_message_file_path)
            language_name = LANGUAGE_NAMES.get(detected_language, detected_language)
            
            logger.info(f"Detected language: {detected_language}")
            logger.info(f"Transcription: {transcription}")
            
            if detected_language == 'en':
                translated_text = self.translate_to_khmer(transcription)
                speech = self.khmer_text_to_speech(translated_text)
                output_file = "khmer_speech.wav"
            else:
                translated_text = self.translate_to_english(transcription)
                if isinstance(translated_text, list):
                    translated_text = translated_text[0] if translated_text else ""
                translated_text = translated_text.strip("[]'")
                speech = self.english_text_to_speech(translated_text)
                output_file = "english_speech.wav"
            
            scipy.io.wavfile.write(output_file, self.mms_model.config.sampling_rate, speech)
            
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message.message_id)
            await update.message.reply_text(f"Original ({language_name}): {transcription}\n\nTranslated: {translated_text}")
            await update.message.reply_voice(voice=open(output_file, "rb"))
            
        except Exception as e:
            logger.error(f"Error processing voice message: {str(e)}", exc_info=True)
            await update.message.reply_text("An error occurred while processing your voice message. Please try again later.")

    async def handle_text(self, update: Update, context):
        message = update.message.text
        processing_message = await update.message.reply_text("Processing your text...")
        
        try:
            english_text = self.translate_to_english(message)
            english_text = english_text[0].strip() if isinstance(english_text, list) else english_text.strip()
            logger.info(f"Translate KH translation: {english_text}")
            
            english_speech = self.english_text_to_speech(english_text)
            output_file = "english_speech.wav"
            scipy.io.wavfile.write(output_file, self.english_tts_model.config.sampling_rate, english_speech)
            
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=processing_message.message_id)
            await update.message.reply_text(f"Original (Khmer): {message}\nTranslated (English): {english_text}")
            await update.message.reply_voice(voice=open(output_file, "rb"))
            
        except Exception as e:
            logger.error(f"Error processing text message: {str(e)}", exc_info=True)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=processing_message.message_id)
            await update.message.reply_text("An error occurred while processing your message. Please try again later.")

    def translate(self, text, src_lang, tgt_lang):
        url = "https://translatekh.mptc.gov.kh/api"
        username = self.translate_kh_username
        password = self.translate_kh_password
        
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json"
        }   
        data = {
            "input_text": [text],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
   
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "translate_text" in result and len(result["translate_text"]) > 0:
                return result["translate_text"][0]
            else:
                logger.error(f"Empty or invalid translation result: {result}")
                raise ValueError("Translation result is empty or invalid")
        except requests.RequestException as e:
            logger.error(f"Error calling Translate KH API: {str(e)}")
            raise

    def translate_to_khmer(self, text):
        return self.translate(text, "eng", "kh")
    
    def translate_to_english(self, text):
        return self.translate(text, "kh", "eng")

    @torch.no_grad()
    def khmer_text_to_speech(self, text):
        inputs = self.khmer_tts_tokenizer(text, return_tensors="pt").to(self.device)
        output = self.mms_model(**inputs).waveform
        return output.squeeze().cpu().numpy()

    @torch.no_grad()
    def english_text_to_speech(self, text):
        inputs = self.english_tts_tokenizer(text, return_tensors="pt").to(self.device)
        output = self.english_tts_model(**inputs).waveform
        return output.squeeze().cpu().numpy()

def main():
    bot = TransformKHBot()
    bot.run()

if __name__ == "__main__":
    main()
