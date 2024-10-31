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

DURATION = 14
STEP_SIZE = 1
CHUNK_SIZE = 512
CHANNELS = -1  # auto-detected
RATE = -1  # auto-detected

next_recording_done_time = 0
next_recording_start_time = 0


def record():
    global audio_queue
    global next_recording_done_time
    global next_recording_start_time
    global RATE
    global CHANNELS
    active_recordings = []
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

        next_counter = 0

        def callback(in_data, frame_count, time_info, status):
            global audio_queue
            global next_recording_done_time
            global next_recording_start_time
            nonlocal next_counter
            current_time = time_info["current_time"]

            if current_time > next_recording_start_time:
                active_recordings.append((int(next_counter), bytearray()))
                next_counter += 1
                next_recording_start_time = current_time + STEP_SIZE
                if next_recording_done_time == 0:
                    next_recording_done_time = current_time + DURATION

            for _, recording in active_recordings:
                recording.extend(in_data)

            if current_time > next_recording_done_time:
                next_recording_done_time = current_time + STEP_SIZE
                done_buffer = active_recordings.pop(0)
                audio_queue.put(done_buffer)

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
    whisper_rate = 16000  # hardcoded from whisper

    last_counter = -1
    while True:
        while audio_queue.empty():
            time.sleep(0.01)

        counter, audio = audio_queue.get()

        assert counter == last_counter + 1
        last_counter = counter

        # Quick catchup for when the model stalls
        if audio_queue.qsize() > 3:
            audio_queue.task_done()
            continue

        # Format sound buffer similar to what whisper do
        audio = numpy.frombuffer(audio, numpy.int16).flatten().astype(numpy.float32) / 32768.0
        # Reduce to 1-channel input to be compatible with whisper
        audio = audio.reshape(-1, CHANNELS)
        audio = audio.mean(axis=1)
        # Reduce to 16k hz to be compatible with whisper
        audio = scipy.signal.resample(audio, int(len(audio) * whisper_rate / RATE))

        audio = whisper.pad_or_trim(audio)
        result = model.transcribe(audio, task="translate", language="en")
        text = result["text"]

        subtitle_queue.put((counter, text))
        audio_queue.task_done()


def update_label():
    global subtitle_queue
    all_words = []
    all_index = -1

    last_counter = -1
    while True:
        counter, text = subtitle_queue.get()

        assert counter >= last_counter + 1
        last_counter = counter

        print("item " + str(counter))
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
