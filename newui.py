import torch
import gradio as gr
import webbrowser

import modules.commons as commons
import modules.utils as utils
from modules.models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text

net_g = None
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
hps = utils.get_hparams_from_file("./configs/config.json")

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    print([f"{p}{t}" for p, t in zip(phone, tone)])
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language



def infer(text, language, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    language = language_marks[language]
    bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
    )
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    return (hps.data.sampling_rate, audio)

def unmodel():
    global net_g
    # Unload Generator
    del net_g  # 释放模型实例
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    print("卸载模型成功了吗？")
	
    # 更新UI上的文本元素
    btn2.update("text", "模型已成功卸载！")

# Load Generator

def loadmodel():
    global net_g
      
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(
        utils.latest_checkpoint_path("./OUTPUT_MODEL/44k"), net_g, None, skip_optimizer=True
    )
    print("加载模型成功了吗？")
    btn3.update("text", "模型已成功加载！")


language_marks = {
    "简体中文": "ZH",
    "日语": "JA",
    "English": "EN",
}
lang = ["简体中文", "日语", "English"]

if __name__ == "__main__":
      
    speaker_ids = hps.data.spk2id
    speakers = list(hps.data.spk2id.keys())
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(
                        label="Text",
                        placeholder="Type your sentence here",
                        value="你是个什么东西！",
                        elem_id=f"tts-input",
                    )
                    # select character
                    char_dropdown = gr.Dropdown(
                        choices=speakers, value=speakers[0], label="character"
                    )
                    language_dropdown = gr.Dropdown(
                        choices=lang, value=lang[0], label="language"
                    )
                    with gr.Row():
                        sdp_ratio_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01,
                            label="SDP Ratio",
                        )
                        noise_scale_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.667,
                            step=0.01,
                            label="Noise Scale",
                        )
                    with gr.Row():
                        noise_scale_w_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                            label="Noise Scale W",
                        )
                        length_scale_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Length Scale",
                        )
                with gr.Column():
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(
                        infer,
                        inputs=[
                            textbox,
                            language_dropdown,
                            sdp_ratio_slider,
                            noise_scale_slider,
                            noise_scale_w_slider,
                            length_scale_slider,
                            char_dropdown,
                        ],
                        outputs=[audio_output],
                    )
            with gr.Column():
                    btn2 = gr.Button("卸载模型!")
                    btn2.click(unmodel,)

            with gr.Column():
                    btn3 = gr.Button("加载模型!")
                    btn3.click(loadmodel,) 
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=True)
