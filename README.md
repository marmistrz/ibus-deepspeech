This is a remake of the original ibus-deepspeech engine to use webrtcvad and deepspeech for speech recognition with IBus.

It's still very buggy!

## Why?
* the webrtcvad approach from Mozilla has much lower CPU usage and, consequently, latency

## Credits
which was the base for the recogntion code
* Mozilla, for [DeepSpeech-examples]
* Elleo, for [the original project]

[the original project]: https://github.com/Elleo/ibus-deepspeech
[DeepSpeech-examples]: https://github.com/mozilla/DeepSpeech-examples

## Alternatives
You can use `mic_vad_streaming.py` from [DeepSpeech-examples] with the `-k` option.
