pip install -U openai-whisper

pip install git+https://github.com/openai/whisper.git

pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

scoop install ffmpeg

pip install setuptools-rust

whisper audio.flac audio.mp3 audio.wav --model medium

whisper japanese.wav --language Japanese

whisper --help

import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])

