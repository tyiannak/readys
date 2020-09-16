import wave


def get_wav_sample_rate(wav_path):
    """
    returns the sampling rate of a WAV file
    :param wav_path: path to the WAV file
    :return: sampling rate in Hz
    """
    with wave.open(wav_path, "rb") as wave_file:
        fs = wave_file.getframerate()
    return fs


# TODO Silence removal will go here
"""
def silence_removal(audio_path):
    audio, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad(int(1))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # Segmenting the Voice audio and save it in list as bytes
    concataudio = [segment for segment in segments]
    joinedaudio = b"".join(concataudio)
    write_wave("Non-Silenced-Audio.wav", joinedaudio, sample_rate)
"""