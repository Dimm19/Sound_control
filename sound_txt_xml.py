
import os
import re
import torch
import whisper
import librosa
import numpy as np
import noisereduce as nr
from torchaudio.functional import resample
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

class AudioTranscriber:
    def __init__(self, model_size="large-v3", device=None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.model = whisper.load_model(model_size, device=self.device)
        self.sentence_endings = {'.', '!', '?', '…'}

    def preprocess_audio(self, audio_path, target_sr=16000):
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            if sr != target_sr:
                y = resample(torch.from_numpy(y), sr, target_sr).numpy()
                sr = target_sr

            y_clean = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=True,
                prop_decrease=0.85,
                n_fft=1024,
                win_length=512,
                use_tqdm=False
            )

            peak = np.max(np.abs(y_clean))
            y_normalized = y_clean * (0.9 / peak) if peak > 0 else y_clean

            trimmed, _ = librosa.effects.trim(
                y_normalized,
                top_db=20,
                frame_length=1024,
                hop_length=256
            )

            return trimmed, sr
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {str(e)}")

    def transcribe(self, audio_path, language="ru", **kwargs):
        try:
            y, sr = self.preprocess_audio(audio_path)

            transcribe_params = {
                "language": language,
                "task": "transcribe",
                "temperature": 0.0,
                "best_of": 1,
                "beam_size": 5,
                "word_timestamps": True,
                "suppress_tokens": [],
                "fp16": False if self.device == "cpu" else True,
                "initial_prompt": "Вот расшифровка с правильной пунктуацией: ",
                **kwargs
            }

            result = self.model.transcribe(y.astype(np.float32), **transcribe_params)

            # Полный текст с пунктуацией для текстового файла
            full_text = result["text"]

            # Фразы с таймстампами для XML (без знаков препинания)
            phrases = self._extract_phrases_with_timestamps(result["segments"])

            return {
                "text": full_text,  # Текст с пунктуацией
                "language": result["language"],
                "phrases": phrases  # Фразы без пунктуации для XML
            }
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def _extract_phrases_with_timestamps(self, segments):
        phrases = []
        current_phrase = []
        phrase_start = None
        phrase_end = None

        for segment in segments:
            for word_info in segment.get("words", []):
                word = word_info["word"].strip()

                # Проверяем, содержит ли слово знак конца предложения
                has_sentence_end = any(punct in word for punct in self.sentence_endings)

                # Очищаем слово от знаков препинания
                clean_word = re.sub(r'[^\w\s]', '', word)
                if not clean_word:
                    continue

                if not current_phrase:
                    phrase_start = word_info["start"]

                current_phrase.append(clean_word)
                phrase_end = word_info["end"]

                # Если слово содержит знак конца предложения - завершаем фразу
                if has_sentence_end:
                    if current_phrase:
                        phrases.append({
                            "words": current_phrase,
                            "start": phrase_start,
                            "end": phrase_end
                        })
                    current_phrase = []
                    phrase_start = None
                    phrase_end = None

        # Добавляем последнюю фразу, если она есть
        if current_phrase:
            phrases.append({
                "words": current_phrase,
                "start": phrase_start,
                "end": phrase_end
            })

        return phrases

    @staticmethod
    def create_xml_output(phrases, output_path):
        root = Element("FoolTextRecognition")
        channels = SubElement(root, "Channels")
        channel = SubElement(channels, "Channel", ChannelNum="0")

        for phrase in phrases:
            start_ms = int(phrase["start"] * 1000)
            end_ms = int(phrase["end"] * 1000)

            for word in phrase["words"]:
                SubElement(channel, "Word",
                          StartMs=str(start_ms),
                          EndMs=str(end_ms),
                          Text=word)

        xml_str = tostring(root, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ", encoding='utf-8')

        with open(output_path, 'wb') as f:
            f.write(pretty_xml)

    @staticmethod
    def supported_formats():
        return [".mp3", ".wav", ".ogg", ".flac"]


def process_file(input_path, output_dir=None, model_size="large-v3"):
    if not os.path.exists(input_path):
        return {"error": "File not found"}

    try:
        transcriber = AudioTranscriber(model_size=model_size)
        result = transcriber.transcribe(input_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(base_name)[0]

            # Сохраняем текстовый файл с пунктуацией
            txt_path = os.path.join(output_dir, f"{name_without_ext}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            # Сохраняем XML файл без пунктуации
            xml_path = os.path.join(output_dir, f"{name_without_ext}.xml")
            transcriber.create_xml_output(result["phrases"], xml_path)

            result["output_files"] = {
                "text": txt_path,
                "xml": xml_path
            }

        return result

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    input_file = "D:/1/1312_0000009762_20250313193848.mp3"  # Используйте / вместо \
    output_folder = "D:/output"

    print("Starting transcription...")
    result = process_file(input_file, output_dir=output_folder, model_size="large-v3")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nTranscription result:")
        print("=" * 50)
        print(result["text"])  # Текст с пунктуацией
        print("=" * 50)
        if "output_files" in result:
            print(f"\nText file with punctuation saved to: {result['output_files']['text']}")
            print(f"XML file without punctuation saved to: {result['output_files']['xml']}")