import whisper
import os
import argparse
import torch
from tqdm import tqdm


def transcribe_one(audio_paths):
    # load audio and pad/trim it to fit 30 seconds
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    audios_mel = [
        whisper.log_mel_spectrogram(
            whisper.pad_or_trim(whisper.load_audio(audio_path)),
            device=model.device,
        )
        for audio_path in audio_paths
    ]
    mels = torch.stack(audios_mel, dim=0)
    # make log-Mel spectrogram and move to the same device as the model

    # detect the spoken language
    _, probs = model.detect_language(mels)
    lang = [max(x, key=x.get) for x in probs]
    # decode the audio
    options = whisper.DecodingOptions(beam_size=3)
    result = whisper.decode(model, mels, options)
    result_text = [x.text for x in result]
    # print the recognized text
    return lang, result_text


"""
_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", default="CJE")
    parser.add_argument("-w", "--whisper_size", default="small")
    parser.add_argument("-o", "--overwrite", default=True)
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            "zh": "[ZH]",
            "ja": "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            "zh": "[ZH]",
            "ja": "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            "zh": "[ZH]",
        }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)
    model = whisper.load_model(args.whisper_size, download_root="./download")
    output_dir = "./dataset"

    def dataloader(speaker):
        cnt = 0
        batch_size = 2
        result = []
        speaker_dir = os.path.join(output_dir, speaker)
        for i, wavfile in tqdm(
            enumerate(os.listdir(speaker_dir)),
            ncols=200,
            colour="green",
            desc="Process {}:".format(speaker),
        ):
            wav_path = os.path.join(speaker_dir, wavfile)
            cnt += 1
            result.append(wav_path)
            if cnt == batch_size:
                cnt = 0
                yield result
                result = []
        if cnt > 0:
            yield result

    speaker_names = os.listdir(output_dir)
    # transcribe text
    for speaker in speaker_names:
        anno_path = os.path.join(output_dir, f"anno_{speaker}.txt")
        if os.path.exists(anno_path) and not args.overwrite:
            continue
        speaker_annos = []
        for wav_paths in dataloader(speaker):
            langs, texts = transcribe_one(wav_paths)
            for i in range(len(wav_paths)):
                if langs[i] not in list(lang2token.keys()):
                    tqdm.write(f"{langs[i]} not supported, ignoring\n")
                    continue
                lang, text = langs[i], texts[i]
                speaker_annos.append(wav_paths[i] + "|" + speaker + "|" + lang +  "|" + text + "\n")

        # write into annotation
        if len(speaker_annos) == 0:
            tqdm.write(
                "Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links."
            )
            tqdm.write(
                "this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported."
            )
        with open(anno_path, "w", encoding="utf-8") as f:
            for line in speaker_annos:
                f.write(line)
