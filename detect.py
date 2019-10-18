from pydub import AudioSegment
import wave
import numpy as np
import pyaudio
from pydub.utils import make_chunks
from pylab import *
import struct


border = 0.3 #音量の閾値

def filter(input):
    fs = 44100
    data = input
    data = frombuffer(data, dtype="int16") / 32768.0

    if max(data) >= border:
        print(max(data))


if __name__ == "__main__":
    
    ### ここはmp3   
    """
    mp3_version = AudioSegment.from_mp3("pinpon.mp3")

    CHUNK = 1024

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(mp3_version.sample_width),
                    channels=mp3_version.channels,
                    rate=mp3_version.frame_rate,
                    output=True)

    for chunk in make_chunks(mp3_version, 16):
        filter(chunk._data)
        stream.write(chunk._data)  


    # ファイルが終わったら終了処理
    stream.stop_stream()
    stream.close()

    p.terminate() 
"""
    ###　ここからマイク
    ma = 0.5
    
    kkk = time.time()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1        #モノラル
    RATE = 44100        #サンプルレート
    CHUNK = 2**11       #データ点数
    RECORD_SECONDS = 10 #録音する時間の長さ
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()


    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            input_device_index=1,   #デバイスのインデックス番号
            frames_per_buffer=CHUNK)
    print ("recording...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        filter(data)
        frames.append(data)
    print ("finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()
