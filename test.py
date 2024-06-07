import ChatTTS
from IPython.display import Audio
import torchaudio
import torch

chat = ChatTTS.Chat()
chat.load_models(compile=False) # 设置为True以获得更快速度

texts = ["你会[laugh]掉下去[laugh]造成严重外伤[laugh]，大量内出血[uv_break]与多处复杂性骨折，也有机会在下方毒雾中[uv_break]受到电击[laugh]或被分解[laugh]。"]

wavs = chat.infer(texts, use_decoder=True)

torchaudio.save(uri="output1.wav", src=torch.from_numpy(wavs[0]), sample_rate=24000, format="wav")
