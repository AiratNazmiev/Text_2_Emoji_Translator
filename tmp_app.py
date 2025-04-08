import torch
import transformers
import re
import streamlit as st
import functools
import os
import multiprocessing
import warnings

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

st.set_page_config(
    page_title="Message to Emoji Translator",
    page_icon=":fast_forward:",
    layout="centered",
)

def translate(text: str, model, tokenizer) -> str:
    input_tokens = tokenizer(text, return_tensors="pt")
    output_tokens = model.generate(**input_tokens)[0]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)

### load translators:
@st.cache_resource
def load_ru_en_translator():
    ru_en_tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    ru_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    ru_en_translator = functools.partial(translate, model=ru_en_model, tokenizer=ru_en_tokenizer)
    
    return ru_en_translator

ru_en_translator = load_ru_en_translator()

st.text(ru_en_translator('Привет!'))
