"""
vad.py — Voice Activity Detection previa a la transcripción.

Por qué existe este módulo
──────────────────────────
Whisper alucina cuando le pasas silencio o ruido. Produce frases fantasma que
nunca se dijeron ("Gracias por ver el video", "Subtítulos por la comunidad"),
porque el decoder siempre genera *algo*: no tiene forma de emitir "aquí no
hubo nada". La única defensa real es no darle audio sin voz.

`faster-whisper` acepta `vad_filter=True`, pero `mlx_whisper.transcribe()` NO,
y MLX es el backend que corre en Apple Silicon. Es decir: en Mac la app corría
sin ningún filtro. Este módulo mueve el VAD *fuera* del modelo, para que ambos
backends reciban exactamente el mismo audio ya filtrado.

Reusa el Silero ONNX que faster-whisper trae empaquetado en sus assets, así que
no agrega dependencias.

Los tres parámetros que importan
────────────────────────────────
Un VAD crudo es inservible por dos razones opuestas: se come la primera sílaba
(cuando detecta voz, ya pasó) y trocea las pausas naturales de respiración.
Se corrigen con tres controles temporales:

  min_speech_ms  (onset)    Descarta tramos de voz más cortos que N ms.
                            Mata clics, golpes de mesa, portazos.
  speech_pad_ms  (prefill)  Agrega N ms a cada lado del tramo detectado.
                            Recupera la sílaba inicial que ya había pasado.
  min_silence_ms (hangover) Espera N ms de silencio antes de cerrar un tramo.
                            No corta cuando respiras a mitad de frase.
"""

import numpy as np

try:
    from faster_whisper.vad import (
        VadOptions,
        get_speech_timestamps,
        collect_chunks,
        get_vad_model,
    )
    _VAD_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depende del entorno
    VadOptions = None
    _VAD_IMPORT_ERROR = e

# Silero procesa el audio en ventanas de 512 muestras (32 ms a 16 kHz).
FRAME_SAMPLES = 512

_model = None
_model_lock = None


def _get_model():
    """Instancia perezosa y compartida del modelo Silero (thread-safe)."""
    global _model, _model_lock
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    with _model_lock:
        if _model is None:
            _model = get_vad_model()
    return _model


def probabilities(audio):
    """Probabilidad de voz por frame, en crudo.

    `get_speech_timestamps` colapsa esto a un sí/no contra un umbral y descarta
    el resto. Conservar la distribución permite distinguir dos situaciones que
    de otro modo se ven idénticas: "no había absolutamente nada" contra
    "había voz que no alcanzó el umbral por poco".

    Devuelve un array float32 de forma (n_frames,).
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size < FRAME_SAMPLES:
        return np.zeros(0, dtype=np.float32)
    # __call__ exige múltiplo exacto de FRAME_SAMPLES
    usable = (audio.size // FRAME_SAMPLES) * FRAME_SAMPLES
    probs = _get_model()(audio[:usable], num_samples=FRAME_SAMPLES)
    return np.asarray(probs, dtype=np.float32).flatten()


# Defaults. min_silence y speech_pad conservan los valores que ya usaba
# la rama de faster-whisper, para no cambiar el comportamiento en Windows.
DEFAULTS = {
    "vad_enabled":        True,
    "vad_threshold":      0.65,  # calibrado: 0/105 ruido colado, 0/17 habla perdida
    "vad_min_speech_ms":  200,   # onset (calibrado, ver abajo)
    "vad_speech_pad_ms":  200,   # prefill
    "vad_min_silence_ms": 300,   # hangover
}


def is_available():
    """True si el modelo Silero y onnxruntime están disponibles."""
    return VadOptions is not None


def unavailable_reason():
    return str(_VAD_IMPORT_ERROR) if _VAD_IMPORT_ERROR else None


def _options_from_settings(settings, threshold=None):
    if threshold is None:
        threshold = float(settings.get("vad_threshold", DEFAULTS["vad_threshold"]))
    return VadOptions(
        threshold=float(threshold),
        min_speech_duration_ms=int(
            settings.get("vad_min_speech_ms", DEFAULTS["vad_min_speech_ms"])
        ),
        min_silence_duration_ms=int(
            settings.get("vad_min_silence_ms", DEFAULTS["vad_min_silence_ms"])
        ),
        speech_pad_ms=int(
            settings.get("vad_speech_pad_ms", DEFAULTS["vad_speech_pad_ms"])
        ),
    )


def analyze(audio, settings, sample_rate=16000, threshold=None):
    """Corre Silero sobre `audio` y devuelve (audio_de_voz, stats).

    `audio` es float32 mono a `sample_rate`.

    Devuelve:
      speech  np.ndarray float32 solo con los tramos de voz. Puede quedar vacío.
      stats   dict con: segments, original_s, speech_s, ratio, ok, reason
    """
    stats = {
        "segments":   0,
        "original_s": len(audio) / float(sample_rate),
        "speech_s":   0.0,
        "ratio":      0.0,
        "ok":         True,
        "reason":     "",
    }

    if not is_available():
        stats["ok"] = False
        stats["reason"] = "silero no disponible: %s" % unavailable_reason()
        return audio, stats

    try:
        opts = _options_from_settings(settings, threshold)
        chunks = get_speech_timestamps(audio, opts, sampling_rate=sample_rate)
    except Exception as e:
        stats["ok"] = False
        stats["reason"] = "error de VAD: %s" % e
        return audio, stats

    if not chunks:
        return np.zeros(0, dtype=np.float32), stats

    # collect_chunks devuelve (lista_de_arrays, metadata), no un array.
    audio_chunks, _meta = collect_chunks(audio, chunks, sampling_rate=sample_rate)
    audio_chunks = [c for c in audio_chunks if c.size]
    if audio_chunks:
        speech = np.concatenate(audio_chunks).astype(np.float32).flatten()
    else:
        speech = np.zeros(0, dtype=np.float32)

    stats["segments"] = len(chunks)
    stats["speech_s"] = len(speech) / float(sample_rate)
    stats["ratio"] = (
        stats["speech_s"] / stats["original_s"] if stats["original_s"] > 0 else 0.0
    )
    return speech, stats


# ─────────────────────────────────────────────────────────────────────────────
#  POLÍTICA DE DECISIÓN
# ─────────────────────────────────────────────────────────────────────────────
#
# La política es estricta: si el VAD no encuentra voz, NO se transcribe. Esa
# decisión no es por simplicidad, es lo que aguantó la medición. Queda el
# registro de lo que se probó y por qué se descartó, para no repetirlo.
#
# Mediciones sobre 103 muestras de no-habla (ruido blanco de 20 semillas por 5
# amplitudes, ruido rosa, música, zumbido de 60 Hz, aire acondicionado, tecleo,
# portazos, silencio) y habla desde 100% hasta 2% de amplitud, limpia y
# enterrada en ruido:
#
#   Silero es casi invariante a la amplitud para habla limpia. Al 2% de
#   volumen puntúa 0.992, igual que a volumen normal, y >80% de sus frames
#   superan 0.3. El miedo a "perder susurros" no está respaldado por los datos.
#
#   El máximo de probabilidad NO sirve como criterio. Es un solo frame entre
#   miles, el estadístico más expuesto a outliers. El ruido blanco alcanzó
#   picos de 0.625, por encima incluso del umbral normal de 0.5.
#
#   Lo que sí separa es la voz SOSTENIDA, y de eso se encarga
#   `min_speech_duration_ms` (el onset) dentro de get_speech_timestamps: exige
#   que la probabilidad se mantenga alta, no que pique una vez.
#
# El onset está calibrado, no elegido a ojo. Barrido contra 100 muestras de
# ruido y 6 frases cortas reales ("sí", "no", "ok", "dale", "listo",
# "gracias", de 0.33 s a 0.73 s):
#
#     onset    falsos positivos / 100      frases cortas detectadas
#     120 ms            2                          6/6
#     160 ms            1                          6/6
#     200 ms            1                          6/6   <- elegido
#     250 ms            1                          6/6
#     300 ms            1                          5/6
#     400 ms            1                          4/6
#
# 200 ms queda en el centro de la banda segura: por debajo de 160 vuelven los
# falsos positivos, desde 300 se empiezan a perder frases cortas. Subirlo más
# no compra nada y cuesta dictados de una palabra.
#
# Nota honesta sobre el residuo: queda 1 falso positivo por cada 100 muestras
# de ruido blanco sintético. No se pudo eliminar sin sacrificar frases cortas.
# Es ruido blanco puro a volumen alto, poco representativo de un micrófono
# real, y el costo cuando ocurre es una frase alucinada suelta, no un fallo
# sistemático. Si aparece en uso real, subir `vad_threshold` antes que el onset.
#
# Se implementó y se descartó un nivel de "rescate": ante cero segmentos,
# repetir con umbral 0.25 para recuperar habla marginal. Recuperaba habla con
# SNR bajo, pero al medirlo contra las 103 muestras de no-habla dio 34 falsos
# positivos (33%), uno de ellos fabricando 2.54 s de "voz" a partir de ruido
# blanco. Eso va directo a Whisper y sale como texto inventado, que es
# exactamente el problema que este módulo existe para eliminar. Descartado.
#
# El habla con SNR por debajo de ~0.2 es irrecuperable aquí, y no importa
# mucho: Whisper tampoco produce nada útil con ese audio.
#
# Lo que hace segura a una política estricta no es un umbral más listo, es que
# el fallo sea VISIBLE. Por eso el llamador muestra el estado "sin voz" en el
# overlay en vez de cerrarse en silencio. Un usuario que ve "sin voz" repite la
# frase; uno que no ve nada cree que la app está rota.
#
# Si aun así descarta demasiado en un entorno concreto, `vad_threshold` es
# ajustable en settings. Bajarlo es una decisión consciente del usuario, con el
# diagnóstico del log a la vista, no una heurística que adivina por él.

# Calibración del umbral, con el onset ya fijo en 200 ms. Sobre 105 muestras
# de no-habla, 11 de habla y 6 frases de una palabra:
#
#     umbral   ruido colado   habla perdida   frases cortas perdidas
#     0.50        1/105           0/11               0/6
#     0.60        1/105           0/11               0/6
#     0.65        0/105           0/11               0/6   <- default
#     0.70        0/105           0/11               1/6
#
# 0.65 domina a 0.50: cero falsos positivos sin costo alguno. Bajarlo NO
# recupera habla marginal, solo cuela ruido: a 0.35 los falsos positivos suben
# a 5/100 y la detección de habla marginal sigue en 1 de 6. Por eso la opción
# "Permisivo" de la UI llega hasta 0.50 y no más abajo.

# Umbral para contar un frame como "voz" en el diagnóstico. Solo informativo.
SUSTAINED_PROB = 0.30


def decide(speech, stats, raw_audio, peak, rms, settings=None, sample_rate=16000):
    """Decide qué audio mandar al modelo, o si no mandar nada.

    Devuelve (audio_a_transcribir, motivo) o (None, motivo) para abortar.

    Deliberadamente NO usa `peak` ni `rms` para juzgar si hubo habla. La
    medición mostró que la energía está anti-correlacionada con el habla en los
    casos límite: un golpe de mesa tiene 74 veces el pico de una voz bajita, y
    aun así solo la voz es habla. La energía sirve para detectar un micrófono
    muerto, y de eso ya se encarga el guard de `peak` en el llamador.
    """
    if speech.size:
        return speech, "voz detectada (%d seg, %.0f%% del audio)" % (
            stats["segments"], stats["ratio"] * 100)

    # Sin voz. Se aborta, pero dejando en el log con cuánta holgura, para que
    # ajustar `vad_threshold` sea una decisión informada y no a ciegas.
    try:
        probs = probabilities(raw_audio)
        if probs.size:
            top = float(probs.max())
            sostenidos = int((probs > SUSTAINED_PROB).sum())
            stats["max_prob"] = top
            stats["sustained_frames"] = sostenidos
            diag = "confianza máx %.2f, %d/%d frames sobre %.2f" % (
                top, sostenidos, probs.size, SUSTAINED_PROB)
        else:
            diag = "audio más corto que un frame"
    except Exception as e:
        diag = "sin diagnóstico: %s" % e

    return None, "sin voz (%s)" % diag
