import numpy
import scipy
import whisper
import threading
import queue
import time
import tkinter
import pyaudiowpatch as pyaudio

model = whisper.load_model(name="large-v3-turbo", device="cuda")
subtitle_queue = queue.Queue()
audio_queue = queue.Queue()

WORDS_PER_LINE = 8
LINES = 5

DURATION = 14  # How long window it has back in time, in seconds
STEP_SIZE = 1  # How often we update, in seconds

CHANNELS = -1  # auto-detected
RATE = -1  # auto-detected
CHUNK_SIZE = -1  # auto based on rate


def record():
    global audio_queue
    global DURATION, STEP_SIZE, RATE, CHANNELS, CHUNK_SIZE
    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        RATE = int(default_speakers["defaultSampleRate"])
        CHANNELS = int(default_speakers["maxInputChannels"])
        CHUNK_SIZE = int(RATE * STEP_SIZE)

        buffer = bytearray()
        max_buffer_size = DURATION * RATE * CHANNELS * 2

        def callback(in_data, frame_count, time_info, status):
            global audio_queue
            nonlocal buffer, max_buffer_size

            now = time.time()

            buffer.extend(in_data)
            if len(buffer) > max_buffer_size:
                buffer = buffer[-max_buffer_size:]

            audio_queue.put((now, bytearray(buffer)))

            return in_data, pyaudio.paContinue

        with p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK_SIZE,
                    input=True,
                    input_device_index=default_speakers["index"],
                    stream_callback=callback
                    ):
            while True:
                time.sleep(DURATION)


def translate():
    global audio_queue
    global RATE, CHANNELS
    whisper_rate = 16000  # hardcoded from whisper
    whisper_channels = 1  # hardcoded from whisper

    while True:
        while audio_queue.empty():
            time.sleep(0.01)

        timestamp, audio = audio_queue.get()

        # Quick catchup for when the model stalls
        if audio_queue.qsize() > 3:
            audio_queue.task_done()
            continue

        # yeeted from whisper code on how it wants buffer formatted
        audio = numpy.frombuffer(audio, numpy.int16).flatten().astype(numpy.float32) / 32768.0

        if CHANNELS != whisper_channels:
            audio = audio.reshape(-1, CHANNELS).mean(axis=1)

        if RATE != whisper_rate:
            audio = scipy.signal.resample(audio, int(len(audio) * whisper_rate / RATE))

        audio = whisper.pad_or_trim(audio)
        result = model.transcribe(audio, task="translate", language="en")
        text = result["text"]

        subtitle_queue.put((timestamp, text))
        audio_queue.task_done()


def update_label():
    global subtitle_queue, STEP_SIZE, RATE, CHANNELS
    all_words = []
    all_index = -1

    while True:
        timestamp, text = subtitle_queue.get()
        striped = text.strip()
        words = striped.split(" ")
        if len(striped) == 0:
            empty_line = WORDS_PER_LINE - (len(all_words) % WORDS_PER_LINE)
            all_words += ([" "] * empty_line)
        elif len(all_words) == 0:
            all_words = words[:]
            all_index = 0
        elif len(all_words) > 0:
            curr_index, last_index = None, None
            looked_at = all_words[all_index:]
            for index_a, word in enumerate(words):
                if word in looked_at:
                    index_b = all_words.index(word, all_index)
                    curr_index = index_a
                    last_index = index_b
                    break
            if curr_index is not None:
                all_index = last_index
                all_words = all_words[:last_index] + words[curr_index:]
            elif curr_index is None:
                all_index = len(all_words)
                all_words += words

        current_words = len(all_words)
        max_words = WORDS_PER_LINE * LINES
        if current_words > max_words:
            removing = 0
            while current_words - removing > max_words:
                removing += WORDS_PER_LINE
            all_words = all_words[removing:]
            all_index -= removing

        grouped = [" ".join(all_words[i:i + WORDS_PER_LINE]) for i in range(0, len(all_words), WORDS_PER_LINE)]
        line_text = "\n".join(grouped) + "\n" + ("_______" * WORDS_PER_LINE)

        now = time.time()
        delay = now - (timestamp - STEP_SIZE)
        delay_str = f"{delay:.2f}"
        print("delay: " + delay_str + " seconds")

        label.config(text=line_text)
        subtitle_queue.task_done()


root = tkinter.Tk()
root.title("Subtitles")
root.geometry("800x200")
label = tkinter.Label(root, font=("Consolas", 24), fg="white", bg="black", justify="left")
label.pack(expand=True, fill="both")

threading.Thread(target=record, daemon=True).start()
threading.Thread(target=translate, daemon=True).start()
threading.Thread(target=update_label, daemon=True).start()

root.mainloop()

while True: time.sleep(1)
