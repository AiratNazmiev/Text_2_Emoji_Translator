import torch
import transformers
import re
import streamlit as st
import functools
import os
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"

twitter_magic_number = 280
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

st.set_page_config(
    page_title="Message to Emoji Translator",
    page_icon=":fast_forward:",
    layout="centered",
)

st.markdown(
    """
    <style>
    textarea:disabled {
        opacity: 1 !important;                     
        color: #000000 !important;
        -webkit-text-fill-color: #FFFFFF !important;
        background-color: inherit !important;
        filter: none !important;
    }
    [data-testid="stTextArea"] label {
        opacity: 1 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Message 📝 to Emoji 😎 Translator")

available_languages = {
    "English 🦁" : ("en", "Enter the message to translate...", "English language"), 
    "Russian 🐻" : ("ru", "Введите сообщение для перевода...", "Русский язык"), 
    "Chinese 🐼" : ("zh", "输入要翻译的文本...", "中文")
}
language_abbr = {name : x[0] for name, x in available_languages.items()}
language_placeholder = {name : x[1] for name, x in available_languages.items()}
language_label = {name : x[2] for name, x in available_languages.items()}

language_option = st.selectbox(
    "Select language:",
    available_languages.keys(),
    index=0,
    placeholder="Select language...",
)

text_value = st.text_area(
    label=language_label[language_option],
    placeholder=language_placeholder[language_option],
    max_chars=twitter_magic_number,
    help=f"Let's speak the language of facts 😉. Facts are limited to {twitter_magic_number} chars (twit size)",
    height=150
)

### load translators:
@st.cache_resource
def load_translators():
    ru_en_tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en", use_auth_token=hf_token)
    ru_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en", use_auth_token=hf_token)
    ru_en_tokenizer.save_pretrained("./local_models/opus-mt-ru-en")
    ru_en_model.save_pretrained("./local_models/opus-mt-ru-en")
    
    ru_en_tokenizer = transformers.AutoTokenizer.from_pretrained("./local_models/opus-mt-ru-en", local_files_only=True)
    ru_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("./local_models/opus-mt-ru-en", local_files_only=True)

    zh_en_tokenizer = transformers.AutoTokenizer.from_pretrained("AiratNazmiev/Helsinki-NLP-opus-mt-zh-en-tokenizer", local_files_only=False)
    zh_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("AiratNazmiev/Helsinki-NLP-opus-mt-zh-en-model", local_files_only=False)
    zh_en_tokenizer.save_pretrained("./local_models/opus-mt-zh-en")
    zh_en_model.save_pretrained("./local_models/opus-mt-zh-en")

    zh_en_tokenizer = transformers.AutoTokenizer.from_pretrained("./local_models/opus-mt-zh-en", local_files_only=True)
    zh_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("./local_models/opus-mt-zh-en", local_files_only=True)
    
    def translate(text: str, model, tokenizer) -> str:
        input_tokens = tokenizer(text, return_tensors="pt")
        output_tokens = model.generate(**input_tokens)[0]
        return tokenizer.decode(output_tokens, skip_special_tokens=True)

    ru_en_translator = functools.partial(translate, model=ru_en_model, tokenizer=ru_en_tokenizer)
    zh_en_translator = functools.partial(translate, model=zh_en_model, tokenizer=zh_en_tokenizer)
    
    return ru_en_translator, zh_en_translator

# @st.cache_resource
# def load_translators():
    
#     ru_en_translator = transformers.pipeline(
#        # "translation_ru_to_en", 
#        "translation",
#         model="Helsinki-NLP/opus-mt-ru-en",
#         cache_dir="~/.cache/huggingface/transformers",
#         use_auth_token=True
#     )

#     zh_en_translator = transformers.pipeline(
#         #"translation_zh_to_en", 
#         "translation",
#         model="Helsinki-NLP/opus-mt-zh-en",
#         cache_dir="~/.cache/huggingface/transformers",
#         use_auth_token=True
#     )
    
#     return ru_en_translator, zh_en_translator
###

### create message-to-emoji translator wrapper class:
@st.cache_resource
def load_msg2emoji_translator():
    class Msg2EmojiTranslator:
        def __init__(
            self,
            tokenizer,
            generator,
            device: torch.device
        ) -> None:
            self.device = device
            self.tokenizer = tokenizer
            self.generator = generator.to(self.device)
            
        def translate(self, sentence: str | list[str], sep: str = '', **kwargs) -> torch.Tensor:
            decoded_emojis_list = []
            
            if isinstance(sentence, str):
                sentence = [sentence]

            for s in sentence:
                text_tokens = self.tokenizer(s, return_tensors="pt")
                generated_emoji_tokens = self.generator.generate(text_tokens["input_ids"].to(self.device), **kwargs)
                decoded_emojis = self.tokenizer.decode(generated_emoji_tokens[0].cpu(), skip_special_tokens=True).replace(" ", "")
                decoded_emojis_list.append(decoded_emojis)
                
            return sep.join(decoded_emojis_list)
    
    tokenizer = transformers.BartTokenizer.from_pretrained('AiratNazmiev/text2emoji-tokenizer')
    generator = transformers.BartForConditionalGeneration.from_pretrained('AiratNazmiev/text2emoji-bart-base')
    
    msg2emoji_translator = Msg2EmojiTranslator(
        tokenizer=tokenizer,
        generator=generator,
        device=device
    )
    
    return msg2emoji_translator
###

### text preprocessing
def text_preprocessing(text: str, ru_en_translator, zh_en_translator, language: str = 'en') -> str:
    if language == 'ru':
        text = ru_en_translator(text)
    elif language == 'zh':
        text = zh_en_translator(text)
    
    if len(text) > twitter_magic_number:
        print(f"It's twit translator. The max length of the input is {twitter_magic_number} characters")
        
    text_re = re.split(r"(?<=[.|!|?|\.\.\.])\s+", text.strip())
    
    return text_re
###

###
def main():
    ru_en_translator, zh_en_translator = load_translators()
    #ru_en_translator, zh_en_translator = None, None
    msg2emoji_translator = load_msg2emoji_translator()
    if st.button("Translate"):
        if not text_value:
            st.warning("Please, enter the message 🤔", icon="⚠️")
        else:
            emoji_text = msg2emoji_translator.translate(
                text_preprocessing(text_value, ru_en_translator=ru_en_translator, zh_en_translator=zh_en_translator, language=language_abbr[language_option]),
                sep='',
                num_beams=5, 
                do_sample=True, 
                max_length=20
            )

            st.text_area(
                label="Emoji language",
                placeholder="Translated texts will appear here.",
                disabled=True,
                value=emoji_text,
            )
        
torch.classes.__path__ = []
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
        
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #FFF1;
        text-align: center;
        padding: 15px;
        font-size: 15px;
        color: #FFFFFF;
    }
    </style>
    <div class="footer">
        &copy; Nazmiev Airat 2025 👋

    </div>
""", unsafe_allow_html=True)
