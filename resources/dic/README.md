# Diccionario de español (RLA-ES)

`es.dic` y `es.aff` son el diccionario ortográfico del español del proyecto
**RLA-ES**, versión 2.9.

- Origen: https://github.com/sbosio/rla-es
- Licencia: triple esquema **disjunto**, GPLv3+, LGPLv3+ o MPL 1.1+.
  El usuario elige libremente bajo cuál de las tres lo utiliza. El texto de
  cada una está en `GPLv3.txt`, `LGPLv3.txt` y `MPL-1.1.txt`, y el detalle
  del esquema en `LICENSE-rla-es.md`.

## Para qué se usa aquí

`terms.py` lo consulta como guard léxico antes de corregir terminología. Su
única función es responder "¿esta palabra existe en español?", para no
convertir palabras legítimas en términos de la empresa. Sin ese guard, la
distancia de edición convertiría "clave" en "Claude" y "apio" en "API".

No se modifica ninguno de los dos archivos: se distribuyen tal cual vienen
del proyecto original.
