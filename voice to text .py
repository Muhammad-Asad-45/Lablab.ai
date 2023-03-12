# !pip install -q gradio
# !pip install -q pyChatGPT
# !pip install -q git+https://github.com/openai/whisper.git
# !pip install -q --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy
# # Imports
# import whisper
# import gradio as gr 
# import time
# import warnings
# import torch
# from pyChatGPT import ChatGPT
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# # Defining Variables
# warnings.filterwarnings("ignore")
# secret_token = ""
# model = whisper.load_model("base")
# model.device
# model_id = "stabilityai/stable-diffusion-2"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, 
#                                                    subfolder="scheduler")

# pipe = StableDiffusionPipeline.from_pretrained(model_id, 
#                                                scheduler=scheduler, 
#                                                revision="fp16", 
#                                                torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# # TranscribeFunction
# def transcribe(audio):

#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     _, probs = model.detect_language(mel)

#     # decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#     result_text = result.text

#     # Pass the generated text to Audio
#     chatgpt_api = ChatGPT(secret_token)
#     resp = chatgpt_api.send_message(result_text)
#     out_result = resp['message']

#     out_image = pipe(out_result, height=768, width=768).images[0]

#     return [result_text, out_result, out_image]
# # Gradio Interface
# output_1 = gr.Textbox(label="Speech to Text")
# output_2 = gr.Textbox(label="ChatGPT Output")
# output_3 = gr.Image(label="Diffusion Output")

# gr.Interface(
#     title = 'OpenAI Whisper and ChatGPT ASR Gradio Web UI', 
#     fn=transcribe, 
#     inputs=[
#         gr.inputs.Audio(source="microphone", type="filepath")
#     ],

#     outputs=[
#         output_1,  output_2, output_3
#     ],
#     live=True).launch()
!apt update 
!apt install ffmpeg

!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
!pip install git+https://github.com/openai/whisper.git 
!pip install diffusers==0.2.4
!pip install transformers scipy ftfy
!pip install "ipywidgets>=7,<8"

!apt update 
!apt install ffmpeg

import whisper
import cv2

# loading model
model = whisper.load_model('small')

# loading audio file
audio = whisper.load_audio('prompt.m4a')
# padding audio to 30 seconds
audio = whisper.pad_or_trim(audio)

# generating spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# decoding
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# ready prompt!
prompt = result.text

# adding tips
prompt += 'Try to draw what you see in the picture'
print(prompt)

# generate the image corresponding to the prompt
img_path = "images/" + prompt + ".jpg"
img = cv2.imread(img_path)
cv2.imshow("Image Prompt", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# provide positive feedback to the child
print("Great job! You did it!")

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    revision='fp16',
    torcj_dtype=torch.float16,
    use_auth_token=True
)

pipe = pipe.to("cuda")

with torch.autocast('cuda'):
    image = pipe(prompt)['sample'][0]

import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(prompt)
plt.axis('off')
plt.savefig('result.jpg')
plt.show()