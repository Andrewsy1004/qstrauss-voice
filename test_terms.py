"""
test_terms.py — Verifica la corrección de terminología.

    python test_terms.py

Dos corpus opuestos. El de POSITIVOS son errores que Whisper cometió de verdad
(medidos transcribiendo voz sintética en español con ruido, no inventados). El
de NEGATIVOS son frases legítimas en español elegidas justamente porque caen
cerca de los términos por distancia de edición:

    clave -> Claude   distancia 2
    apio  -> API      distancia 1
    así   -> API      distancia 1

Corromper una de esas es mucho peor que dejar un término sin corregir, así que
un solo fallo en NEGATIVOS invalida la corrida completa.
"""
import json
import os
import sys

import terms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Errores reales observados. El comentario dice de dónde salió cada uno.
POSITIVOS = [
    ("Revisa el boxfront de la cuenta", "Workfront"),        # ruido 0.04
    ("Llama a la API de Antropi", "Anthropic"),              # ruido 0.02
    ("Hicimos el instaje del requerimiento", "intake"),      # ruido 0.06
    ("Revisa el work front de la cuenta", "Workfront"),      # sin prompt
    ("el workfron de la cuenta", "Workfront"),
    ("reunion por microsoft tims", "Microsoft Teams"),
    ("falta el deploiment en produccion", "deployment"),
]

# Español legítimo. Ninguna debe cambiar ni una letra.
NEGATIVOS = [
    "la clave de acceso caduca el viernes",
    "me parece un punto clave para la decisión",
    "compré apio y tomates en el mercado",
    "revisamos los temas pendientes de la reunión",
    "así que hay que renovarla cuanto antes",
    "la función tiene un instante de espera",
    "el área comercial define la versión final",
    "el flujo de aprobación quedó documentado",
    "el asunto del caso quedó en la nube",
    "hicimos el análisis técnico del proyecto",
    "el costo del curso subió este año",
    "mandé el correo con el resumen adjunto",
    "necesito la firma del contrato antes del jueves",
    "la base de datos está en mantenimiento",
    "el equipo entregó el módulo sin documentación",
    "revisé el presupuesto antes de enviarlo",
]


def cargar_terminos():
    p = os.path.join(BASE_DIR, "dictionary.json")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    hints = data.get("hints", [])
    corr = {k.lower(): v for k, v in data.get("corrections", {}).items()}
    return corr, list(dict.fromkeys(hints + list(corr.values())))


def main():
    lex = terms.Lexico()
    if not lex.disponible:
        print("Sin léxico español (NSSpellChecker solo existe en macOS).")
        print("La corrección difusa queda desactivada, que es la degradación segura.")
        return 0

    correcciones, lista = cargar_terminos()
    print("%d términos, %d correcciones exactas\n" % (len(lista), len(correcciones)))

    def pipeline(t):
        t = terms.aplicar_exactas(t, correcciones)
        t, cambios = terms.aplicar_difusas(t, lista, lex)
        return t, cambios

    fallos = 0

    print("=== POSITIVOS: deben corregirse sin comerse palabras vecinas ===")
    ok = 0
    for texto, esperado in POSITIVOS:
        salida, _ = pipeline(texto)
        acerto = esperado.lower() in salida.lower()
        # El n-grama no debe absorber artículos ni preposiciones vecinas.
        entero = len(salida.split()) >= len(texto.split()) - esperado.count(" ") - 1
        bien = acerto and entero
        ok += bien
        print("  %s %-38s -> %s" % ("OK " if bien else ">>>", texto, salida))
    print("  %d de %d" % (ok, len(POSITIVOS)))
    if ok < len(POSITIVOS):
        fallos += 1

    print("\n=== NEGATIVOS: no deben cambiar ni una letra ===")
    rotos = []
    for texto in NEGATIVOS:
        salida, cambios = pipeline(texto)
        if salida != texto:
            rotos.append((texto, salida, cambios))
    print("  %d de %d intactas" % (len(NEGATIVOS) - len(rotos), len(NEGATIVOS)))
    for a, b, c in rotos:
        print("     CORROMPIDO: %s\n              -> %s   %s" % (a, b, c))
    if rotos:
        print("     FALLA CRITICA: se corrompio español legítimo")
        fallos += 1

    print("\n%s" % ("TODO OK" if fallos == 0 else "%d GRUPO(S) CON FALLA" % fallos))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
