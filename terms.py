"""
terms.py — Corrección de terminología con guard léxico.

El problema que resuelve
────────────────────────
El diccionario listaba a mano cada variante que Whisper podía inventar
("workfron", "work front", "work fron"). No escala: siempre aparece una cuarta.

La solución obvia, distancia de edición contra los términos, es una trampa en
español. Medido sobre errores reales de Whisper:

    instaje  -> intake      distancia 2   hay que corregirlo
    clave    -> Claude      distancia 2   JAMÁS tocarlo
    apio     -> API         distancia 1   JAMÁS tocarlo
    así      -> API         distancia 1   JAMÁS tocarlo

Las distribuciones se solapan por completo: no existe umbral que las separe.
La distancia de edición no sabe qué es una palabra real, y ese es justamente
el conocimiento que hace falta.

La solución
───────────
Antes de medir distancias, se consulta un léxico español y solo se consideran
candidatas las palabras que NO existen en español. Con eso, "clave", "apio" y
"así" quedan fuera de juego antes de que la distancia importe.

Un detalle mata la versión ingenua de esa idea: si Whisper se come una tilde,
"así" queda como "asi", el léxico dice "no existe" y la distancia 1 la
convertiría en "API". Por eso, cuando una palabra no existe, se le piden
sugerencias al corrector: si alguna sugerencia es la misma palabra con tildes,
es un desliz de acento y no un término desconocido. Verificado sobre "asi",
"funcion", "reunion", "area", "version" y "manana": los seis quedan protegidos,
mientras "instaje", "antropi", "boxfront", "workfron" y "qstrauss" siguen
siendo candidatos.

Disponibilidad
──────────────
El léxico viene de NSSpellChecker, que solo existe en macOS. Sin léxico NO se
hace corrección difusa: se cae a la coincidencia exacta de siempre. Es la
degradación segura, porque el modo de falla de esta función es corromper texto
correcto, y eso es peor que no corregir nada.
"""

import re
import unicodedata

# Se comparan palabras de largo similar. Un candidato mucho más corto o mucho
# más largo que el término casi nunca es el mismo término mal oído.
MAX_LEN_RATIO = 0.34
# Distancia máxima permitida, en proporción al largo de la palabra más larga.
MAX_DIST_RATIO = 0.34
# Palabras muy cortas se descartan: en ellas cualquier distancia es enorme en
# proporción, y son las que más falsos positivos producen.
MIN_LEN = 4
# Ventana de n-gramas, para el caso "work front" -> "Workfront".
MAX_NGRAM = 3

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def sin_tildes(s):
    """Minúsculas y sin diacríticos. 'Función' -> 'funcion'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def clave(s):
    """Clave de comparación: sin tildes, sin puntuación, solo alfanuméricos."""
    return "".join(c for c in sin_tildes(s) if c.isalnum())


def levenshtein(a, b):
    """Distancia de edición. Implementación propia a propósito.

    El corpus son ~12 términos contra ~30 palabras por transcripción, así que
    una dependencia con extensión en C (rapidfuzz) agregaría peso y riesgo de
    empaquetado en PyInstaller sin ganancia medible.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previa = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        actual = [i]
        for j, cb in enumerate(b, 1):
            actual.append(min(
                previa[j] + 1,          # borrar
                actual[j - 1] + 1,      # insertar
                previa[j - 1] + (ca != cb),  # sustituir
            ))
        previa = actual
    return previa[-1]


# ─── Léxico ──────────────────────────────────────────────────────────────────

class Lexico(object):
    """Envuelve NSSpellChecker. `disponible` es False fuera de macOS."""

    def __init__(self, idioma="es"):
        self.idioma = idioma
        self.disponible = False
        self._cache = {}
        self._checker = None
        try:
            from AppKit import NSSpellChecker
            from Foundation import NSMakeRange
            self._checker = NSSpellChecker.sharedSpellChecker()
            self._rango = NSMakeRange
            if idioma in list(self._checker.availableLanguages() or []):
                self.disponible = True
        except Exception:
            self.disponible = False

    def _bien_escrita(self, palabra):
        r = self._checker.checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
            palabra, 0, self.idioma, False, 0, None)
        return r[0].length == 0

    def _es_desliz_de_tilde(self, palabra):
        """True si el corrector sugiere la misma palabra con tildes.

        Sin esto, 'asi' (por 'así') se ve como término desconocido y la
        distancia 1 la convertiría en 'API'.
        """
        try:
            sug = self._checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
                self._rango(0, len(palabra)), palabra, self.idioma, 0)
        except Exception:
            return False
        objetivo = sin_tildes(palabra)
        for s in list(sug or [])[:8]:
            if sin_tildes(s) == objetivo:
                return True
        return False

    def es_palabra_real(self, palabra):
        """True si existe en español, o si es esa misma palabra sin tilde."""
        if not self.disponible:
            # Sin léxico se asume que todo es real, así nada se corrige.
            return True
        key = palabra.lower()
        if key in self._cache:
            return self._cache[key]
        try:
            ok = self._bien_escrita(palabra) or self._es_desliz_de_tilde(palabra)
        except Exception:
            ok = True  # ante la duda, no tocar
        self._cache[key] = ok
        return ok


# ─── Corrección ──────────────────────────────────────────────────────────────

def _mejor_termino(candidato, terminos_clave, permitir_exacto):
    """(termino, ratio_de_distancia) más cercano al candidato, o (None, None).

    `permitir_exacto` distingue dos situaciones que se ven iguales: para una
    palabra suelta, distancia 0 significa "ya está bien escrita, no tocar";
    para un n-grama, significa "estas palabras juntas SON el término" y hay
    que unirlas. Ese es justo el caso "work front" -> "Workfront".
    """
    if len(candidato) < MIN_LEN:
        return None, None
    mejor, mejor_r = None, None
    for original, ckey in terminos_clave:
        largo = max(len(candidato), len(ckey))
        if abs(len(candidato) - len(ckey)) / float(largo) > MAX_LEN_RATIO:
            continue
        d = levenshtein(candidato, ckey)
        if d == 0 and not permitir_exacto:
            return None, None
        r = d / float(largo)
        if r > MAX_DIST_RATIO:
            continue
        if mejor_r is None or r < mejor_r:
            mejor, mejor_r = original, r
    return mejor, mejor_r


def aplicar_exactas(texto, correcciones):
    """Reemplazos exactos del diccionario. Siempre ganan sobre lo difuso.

    El \\b es obligatorio: sin él, 'aser' -> 'Azure' convierte 'aserrín' en
    'Azurerín' y 'caserío' en 'cAzureío'.
    """
    for mal, bien in correcciones.items():
        texto = re.sub(r"\b%s\b" % re.escape(mal), bien, texto, flags=re.IGNORECASE)
    return texto


def aplicar_difusas(texto, terminos, lexico):
    """Corrige términos mal oídos que no existen como palabra en español."""
    if not terminos or lexico is None or not lexico.disponible:
        return texto, []

    terminos_clave = [(t, clave(t)) for t in terminos if clave(t)]
    tokens = list(_WORD_RE.finditer(texto))
    if not tokens:
        return texto, []

    reemplazos = []   # (inicio, fin, texto_nuevo, motivo)
    usados = set()

    # Para cada posición se evalúan todos los n-gramas y gana el de MEJOR
    # calidad de match, no el más largo. Eso resuelve solo el problema de que
    # un n-grama se trague palabras vecinas: "el work front" contra
    # "workfront" da ratio 0.18, pero "work front" da 0.00, así que el
    # artículo queda fuera sin necesidad de una regla especial para artículos.
    # Una palabra que YA es exactamente un término se bloquea antes de nada.
    # Si no, no genera candidato propio (está bien escrita) y queda libre para
    # que un n-grama vecino se la trague: "el Workfront" contra "workfront" da
    # ratio 0.18 y devolvía "Revisa Workfront de la cuenta", sin el artículo.
    claves_exactas = set(k for _o, k in terminos_clave)
    for idx, m in enumerate(tokens):
        if clave(m.group(0)) in claves_exactas:
            usados.add(idx)

    candidatos = []
    for i in range(len(tokens)):
        for n in range(1, MAX_NGRAM + 1):
            if i + n > len(tokens):
                break
            if any((i + k) in usados for k in range(n)):
                continue
            grupo = tokens[i:i + n]
            palabras = [m.group(0) for m in grupo]

            # Guard léxico: si TODAS existen en español es texto legítimo.
            # Basta una desconocida para sospechar del grupo.
            if all(lexico.es_palabra_real(p) for p in palabras):
                continue

            cand = clave("".join(palabras))
            termino, ratio = _mejor_termino(cand, terminos_clave, n > 1)
            if termino is None:
                continue
            candidatos.append((ratio, n, i, grupo, palabras, termino))

    # Mejor ratio primero; a igualdad, el n-grama más corto.
    candidatos.sort(key=lambda c: (c[0], c[1]))
    for _ratio, n, i, grupo, palabras, termino in candidatos:
        if any((i + k) in usados for k in range(n)):
            continue
        reemplazos.append((grupo[0].start(), grupo[-1].end(), termino,
                           "%s -> %s" % (" ".join(palabras), termino)))
        for k in range(n):
            usados.add(i + k)

    if not reemplazos:
        return texto, []

    reemplazos.sort(key=lambda r: r[0], reverse=True)
    for ini, fin, nuevo, _m in reemplazos:
        texto = texto[:ini] + nuevo + texto[fin:]
    return texto, [r[3] for r in reversed(reemplazos)]
