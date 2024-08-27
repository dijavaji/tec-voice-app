import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
ELEVEN_API_KEY = config["ELEVENLABS_API_KEY"]

#habilitar ambiente source .venv/bin/activate
#instalar pip install -r requirements.txt
#ejecutar app python3 main.py
# 1. Transcribir texto
# Usamos Whisper: https://github.com/openai/whisper
# Alternativa API online: https://www.assemblyai.com
def translator(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")

    print(f"Texto original: {transcription}")
    #2. traducir texto
    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(f"Se ha producido un error traduciendo el texto: {str(e)}")

    #3 genero audio traducido
    client = ElevenLabs(api_key=ELEVEN_API_KEY )
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=en_transcription,
        model_id="eleven_turbo_v2",
        # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    save_file_path = "audios/en.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs= gr.Audio(
        sources=['microphone'],
        type='filepath'
    ),
    outputs=[gr.Audio(label="Ingles")],
    title="Traductor de voz",
    description="Traductor de voz con IA varios Idiomas"
)

web.launch()

if __name__ == '__main__':
    print_hi('PyCharm')