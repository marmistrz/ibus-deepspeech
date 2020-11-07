# This file is mostly based on mic_vad_straming.py example from
# https://github.com/mozilla/DeepSpeech-examples
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
from dataclasses import dataclass
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from scipy import signal
from typing import Optional, Callable


logging.basicConfig(level=20)

DEFAULT_SAMPLE_RATE = 16000


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            # pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            "format": self.FORMAT,
            "channels": self.CHANNELS,
            "rate": self.input_rate,
            "input": True,
            "frames_per_buffer": self.block_size_input,
            "stream_callback": proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs["input_device_index"] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, "rb")

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(), input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate
    )

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                  |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


class DictationThread(threading.Thread):
    def __init__(
        self,
        model_path: str,
        scorer_path: str,
        recognition_callback: Callable[[str], None],
        savewav: bool = False,
        vad_aggressiveness: int = 3,
        device: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.scorer_path = scorer_path
        self.recognition_callback = recognition_callback
        self.savewav = savewav
        self.vad_aggressiveness = vad_aggressiveness
        self.device = device
        self.stopped = False

    def request_stop(self) -> None:
        self.stopped = True

    def run(self) -> None:
        recognition_callback = self.recognition_callback
        # Load DeepSpeech model
        if os.path.isdir(self.model_path):
            model_dir = self.model_path
            self.model_path = os.path.join(model_dir, "output_graph.pb")
            self.scorer_path = os.path.join(model_dir, self.scorer_path)

        print("Initializing model...")
        logging.info("model_path: %s", self.model_path)
        model = deepspeech.Model(self.model_path)
        if self.scorer_path:
            logging.info("scorer_path: %s", self.scorer_path)
            model.enableExternalScorer(self.scorer_path)

        # Start audio with VAD
        vad_audio = VADAudio(
            aggressiveness=self.vad_aggressiveness,
            device=self.device,
            input_rate=DEFAULT_SAMPLE_RATE,
            file=None,
        )
        print("Listening...")
        frames = vad_audio.vad_collector()

        # Stream from microphone to DeepSpeech using VAD
        stream_context = model.createStream()
        wav_data = bytearray()
        for frame in frames:
            if self.stopped:
                return
            if frame is not None:
                logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                logging.debug("end utterence")
                text = stream_context.finishStream()
                recognition_callback(text)
                stream_context = model.createStream()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream from microphone to DeepSpeech using VAD"
    )

    parser.add_argument(
        "-v",
        "--vad_aggressiveness",
        type=int,
        default=3,
        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3",
    )
    parser.add_argument("--nospinner", action="store_true", help="Disable spinner")
    parser.add_argument(
        "-f", "--file", help="Read from .wav file instead of microphone"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)",
    )
    parser.add_argument("-s", "--scorer", help="Path to the external scorer file.")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.",
    )

    ARGS = parser.parse_args()
    callback = lambda str: print("Recognized: %s" % str)
    # if ARGS.savewav:
    #     os.makedirs(ARGS.savewav, exist_ok=True)
    DictationThread(ARGS.model, ARGS.scorer, callback).run()
