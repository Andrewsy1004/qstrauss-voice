"""
test_vad.py — Verifica la política de VAD sin micrófono ni permisos.

    python test_vad.py

Mide dos tasas opuestas:
  falsos positivos  ruido que pasa como voz y terminaría alucinado por Whisper
  falsos negativos  habla real que se descarta y el usuario pierde

El audio de habla son fixtures CONGELADOS en tests/fixtures/. Antes se
generaban con `say` en cada corrida, y eso tenía dos problemas: la generación
varía entre invocaciones, así que el test fallaba 1 de cada 4 veces sin que
cambiara nada del código (un test intermitente enseña a ignorar los fallos); y
`say` solo existe en macOS, así que en Windows la mitad del test se saltaba.
"""
import glob
import json
import os
import sys
import wave

import numpy as np

import vad

SR = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRASES_CORTAS = ["si", "no", "ok", "dale", "listo", "gracias"]
FIXTURES = os.path.join(BASE_DIR, "tests", "fixtures")


def cargar_settings():
    p = os.path.join(BASE_DIR, "settings.json")
    s = dict(vad.DEFAULTS)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            s.update(json.load(f))
    # Los defaults del módulo mandan sobre un settings.json viejo que no
    # tenga las claves de VAD todavía.
    for k, v in vad.DEFAULTS.items():
        s.setdefault(k, v)
    return s


def decide(audio, settings):
    """Corre el pipeline completo tal como lo hace la app."""
    speech, stats = vad.analyze(audio, settings, SR)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    elegido, motivo = vad.decide(speech, stats, audio, peak, rms, settings, SR)
    return elegido is not None, motivo


def muestras_no_habla():
    """Ruido, silencio, música, zumbido, golpes. Nada debería pasar."""
    out = []
    for seed in range(20):
        r = np.random.RandomState(seed)
        for amp in (0.02, 0.08, 0.15, 0.30, 0.5):
            out.append(("ruido s%d a%.2f" % (seed, amp),
                        (r.randn(SR * 5) * amp).astype(np.float32)))
    r = np.random.RandomState(99)
    t = np.arange(SR * 5) / float(SR)
    out.append(("musica", sum(0.12 * np.sin(2 * np.pi * f * t)
                              for f in (261, 329, 392, 523)).astype(np.float32)))
    out.append(("zumbido 60Hz", (0.2 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)))
    out.append(("aire acondicionado",
                (np.convolve(r.randn(SR * 5), np.ones(40) / 40, "same") * 0.3).astype(np.float32)))
    out.append(("silencio", np.zeros(SR * 5, dtype=np.float32)))
    golpe = np.concatenate([
        np.zeros(SR, dtype=np.float32),
        (r.randn(3000) * 0.9 * np.exp(-np.arange(3000) / 300.0)).astype(np.float32),
        np.zeros(SR * 3, dtype=np.float32)])
    out.append(("portazo", golpe))
    return out


def _leer_wav(path):
    w = wave.open(path)
    try:
        raw = w.readframes(w.getnframes())
    finally:
        w.close()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def muestras_habla():
    """Habla real desde fixtures congelados. Mismo corpus en cada corrida."""
    larga = os.path.join(FIXTURES, "larga.wav")
    if not os.path.exists(larga):
        return [], []
    sp = _leer_wav(larga)

    normales = [("habla x%.2f" % k, (sp * k).astype(np.float32))
                for k in (1.0, 0.5, 0.3, 0.15, 0.05, 0.02, 0.01)]
    for d in (1, 3, 10):
        pad = np.zeros(SR * d, dtype=np.float32)
        normales.append(("habla + %ds de silencio" % d, np.concatenate([pad, sp, pad])))
    normales.append(("habla + ruido leve",
                     (sp + np.random.RandomState(3).randn(len(sp)) * 0.01).astype(np.float32)))

    cortas = []
    pad = np.zeros(SR, dtype=np.float32)
    for frase in FRASES_CORTAS:
        p = os.path.join(FIXTURES, "%s.wav" % frase)
        if os.path.exists(p):
            a = _leer_wav(p)
            cortas.append(('"%s" (%.2fs)' % (frase, len(a) / float(SR)),
                           np.concatenate([pad, a, pad]).astype(np.float32)))
    return normales, cortas


def main():
    if not vad.is_available():
        print("Silero no disponible: %s" % vad.unavailable_reason())
        return 1

    settings = cargar_settings()
    print("umbral=%.2f  onset=%dms  pad=%dms  hangover=%dms\n" % (
        settings["vad_threshold"], settings["vad_min_speech_ms"],
        settings["vad_speech_pad_ms"], settings["vad_min_silence_ms"]))

    fallos = 0

    noh = muestras_no_habla()
    fp = [n for n, a in noh if decide(a, settings)[0]]
    print("NO HABLA   %3d muestras  ->  %d falsos positivos" % (len(noh), len(fp)))
    for n in fp:
        print("     paso como voz: %s" % n)
    # 1 por cada 100 es el residuo conocido y documentado en vad.py.
    if len(fp) > max(2, len(noh) // 50):
        print("     FALLA: demasiados falsos positivos")
        fallos += 1

    normales, cortas = muestras_habla()
    if not normales:
        print("\nFaltan los fixtures en tests/fixtures/, se salta la parte de habla")
    else:
        fn = [n for n, a in normales if not decide(a, settings)[0]]
        print("HABLA      %3d muestras  ->  %d falsos negativos" % (len(normales), len(fn)))
        for n in fn:
            print("     se descarto: %s" % n)
        if fn:
            print("     FALLA: se perdio habla real")
            fallos += 1

        if cortas:
            fnc = [n for n, a in cortas if not decide(a, settings)[0]]
            print("CORTAS     %3d muestras  ->  %d descartadas" % (len(cortas), len(fnc)))
            for n in fnc:
                print("     se descarto: %s" % n)
            if fnc:
                print("     FALLA: el onset esta muy alto, se pierden dictados de una palabra")
                fallos += 1

    print("\nmotivo de ejemplo al abortar:")
    print("   %s" % decide(np.zeros(SR * 5, dtype=np.float32), settings)[1])

    print("\n%s" % ("TODO OK" if fallos == 0 else "%d GRUPO(S) CON FALLA" % fallos))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
