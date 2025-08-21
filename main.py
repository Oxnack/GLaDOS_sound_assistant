import json
import queue
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
from TeraTTS import TTS
from ruaccent import RUAccent
import g4f
import time
import re

TTS_MODEL = "TeraTTS/glados2-g2p-vits"
MODEL_PATH = "vosk-model-ru-0.42"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000
RULES_FOR_DEEPSEEK = "веди себя так же как GLaDOS из игр Portal и Portal2, не общайся слишком добро, ты должен максимально быть похожим на оригинальную GLaDOS, отвечай коротко не длинно, отвечай исключительно на русском кириллицей если нужно паписать английское слово например Aperture Scince пиши на кириллице Апертуре Сцайнс, ты не голосовой помошник ты GLaDOS совершенная машина, не будь слишком вежлива, так же учитывай контекст (если его нет значит разговор только начался)"
is_playing = False
context = "Контекст: "

is_glados_speaking = False
audio_lock = threading.Lock()

def get_deepseek_response(prompt: str) -> str:
    global context
    context += "    user Say: " +  prompt
    print(RULES_FOR_DEEPSEEK + context + "Now user say: " + prompt)
    response = g4f.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": RULES_FOR_DEEPSEEK + context + "Now user say: " + prompt}],
        stream=True
    )
    full_response = ""
    for chunk in response:
        full_response += str(chunk)
  
    print("\n\n\n" + "fuuuul --------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>\n" + full_response + "\nfuuuul stop---------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n\n")

    
    clean_response = re.sub(r'.*?<think>.*?</think>', '', full_response, flags=re.DOTALL)
    clean_response = re.sub(r'.*?reasoning....*?Done in .*?s.', '', clean_response, flags=re.DOTALL)
    english_letters = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    for letter in english_letters:
        clean_response = clean_response.replace(letter, "")
    context += "   Your answer: " + clean_response.strip()
    return clean_response.strip()

accentizer = RUAccent()
custom_dict = {
    'ГЛаДОС': 'ГЛ+А+ДОС', 
    'ГЛАДОС': 'ГЛ+АДОС', 
    'ГлаДОС': 'Гл+аДОС',
    'ИИ': '+И-+И',
    'АИ': '+А-+И'
}
accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)
    
tts = TTS(TTS_MODEL, add_time_to_end=1.0, tokenizer_load_dict=True)

def text_to_glados_voice(text, output_file=None):
    """Convert text to GLaDOS voice using TeraTTS"""
    global is_glados_speaking
    
    with audio_lock:
        is_glados_speaking = True
        print("🔇 Запись заблокирована (ГЛаДОС говорит)")
    
    try:
        # Process text and convert to speech
        processed_text = accentizer.process_all(text.strip())
        
        if output_file:
            # Save to file instead of playing
            audio = tts(processed_text, play=False, lenght_scale=1.1)
            tts.save_wav(audio, output_file)
            print(f"Audio saved to: {output_file}")
            return output_file
        else:
            # Try to play (will fail if PortAudio not installed)
            try:
                audio = tts(processed_text, play=True, lenght_scale=1.1)
                return audio
            except:
                print("PortAudio not found. Please install it or use output_file parameter")
                return None
    finally:
        # Разблокируем запись после окончания речи
        with audio_lock:
            is_glados_speaking = False
            print("🔊 Запись разблокирована (ГЛаДОС закончила говорить)")

# ========== НАСТРОЙКИ ==========
MODEL_PATH = "vosk-model-ru-0.42"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

# ========== ИНИЦИАЛИЗАЦИЯ ==========
# print("Загружаем модель Vosk...")
# try:
#     model = Model(MODEL_PATH)
# except Exception as e:
#     print(f"Ошибка загрузки модели: {e}")
#     print("Скачайте модель с https://alphacephei.com/vosk/models")
#     print("и распакуйте в папку с скриптом")
#     sys.exit(1)

# print("Модель загружена успешно!")

# # Создаем распознаватель
# recognizer = KaldiRecognizer(model, SAMPLE_RATE)
# recognizer.SetWords(True)

# # Очереди для обмена данными
# audio_queue = queue.Queue()
# result_queue = queue.Queue()
# is_recording = True

# ========== ФУНКЦИЯ ЗАПИСИ АУДИО ==========
def record_audio():
    """Записывает аудио с микрофона"""
    global is_recording
    
    def audio_callback(indata, frames, time, status):
        """Callback функция для получения аудиоданных"""
        if status:
            print(f"Audio status: {status}")
        
        # Проверяем, не говорит ли сейчас ГЛаДОС
        with audio_lock:
            if is_glados_speaking:
                return  # Пропускаем запись, если ГЛаДОС говорит
        
        # Добавляем raw bytes в очередь только если нет блокировки
        audio_queue.put(bytes(indata))
    
    print("Запуск записи аудио...")
    
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            print("Запись начата. Говорите...")
            while is_recording:
                sd.sleep(100)
                
    except Exception as e:
        print(f"Ошибка записи аудио: {e}")

# ========== ФУНКЦИЯ ОБРАБОТКИ АУДИО ==========
def process_audio():
    """Обрабатывает аудио и распознает речь"""
    print("Запуск обработки аудио...")
    global is_glados_speaking
    while True:
        # Проверяем, не говорит ли сейчас ГЛаДОС
        with audio_lock:
            if is_glados_speaking:
                time.sleep(0.1)
                continue
        
        try:
            # Получаем аудиоданные из очереди
            data = audio_queue.get(timeout=1.0)
        except queue.Empty:
            if not is_recording:
                break
            continue
        
        # Подаем данные в распознаватель
        if recognizer.AcceptWaveform(data):
            # Получаем окончательный результат (когда речь закончилась)
            result = json.loads(recognizer.Result())
            text = result.get('text', '').strip()
            if text:
                print(f"\n✅ РАСПОЗНАНО: {text}")
                is_glados_speaking = True
                llm_response = get_deepseek_response(text)
                text_to_glados_voice(llm_response)
                result_queue.put(text)
        else:
            # Частичный результат (речь продолжается)
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get('partial', '').strip()
            if partial_text:
                # Выводим частичный результат в той же строке
                print(f"\r🎤 Говорите: {partial_text}", end='', flush=True)

# ========== ОСНОВНАЯ ПРОГРАММА ==========
def main():
    global is_recording
    
    print("=" * 60)
    print("VOSK РЕАЛЬНО-ВРЕМЕННОЕ РАСПОЗНАВАНИЕ РЕЧИ")
    print("=" * 60)
    print("Говорите четко в микрофон...")
    print("Для выхода нажмите Ctrl+C")
    print("=" * 60)
    
    # Запускаем потоки
    record_thread = threading.Thread(target=record_audio)
    record_thread.daemon = True
    record_thread.start()
    
    process_thread = threading.Thread(target=process_audio)
    process_thread.daemon = True
    process_thread.start()
    
    try:
        # Главный цикл
        while True:
            sd.sleep(100)
            
    except KeyboardInterrupt:
        print("\n\nЗавершение работы...")
        is_recording = False
        
        # Даем потокам время завершиться
        record_thread.join(timeout=2.0)
        
        # Получаем последний результат если есть
        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get('text', '').strip()
        if final_text:
            print(f"\n✅ ПОСЛЕДНЯЯ ФРАЗА: {final_text}")
            result_queue.put(final_text)
    
    # Выводим все распознанные результаты
    print("\n" + "=" * 60)
    print("ВСЕ РАСПОЗНАННЫЕ ФРАЗЫ:")
    print("=" * 60)
    
    all_results = []
    while not result_queue.empty():
        result = result_queue.get()
        all_results.append(result)
        print(f"• {result}")
    
    if not all_results:
        print("Нет распознанных фраз")
    else:
        print(f"\nВсего распознано фраз: {len(all_results)}")
    
    print("Работа завершена!")

# if __name__ == "__main__":
#     main()