# QStrauss Voice

Dictado por voz que funciona sin conexión. Pulsas un atajo, hablas, y el texto
aparece donde tengas el cursor.

El audio nunca sale de tu equipo: se transcribe localmente y se descarta.

## Instalación

Descarga desde [la página de descargas](https://akiotanaka847.github.io/qstrauss-voice/)
o desde [Releases](https://github.com/akiotanaka847/qstrauss-voice/releases).

**macOS:** arrastra `QStrauss Voice.app` a Aplicaciones. Al primer arranque pide
dos permisos y hacen falta los dos:

- **Micrófono**, para escucharte.
- **Accesibilidad**, para pegar el texto. Sin este permiso la app transcribe
  pero el texto se queda en el portapapeles y tienes que pegarlo tú.

**Windows:** descomprime y ejecuta `QStrauss Voice.exe`. La carpeta completa
tiene que quedar junta, el ejecutable necesita los archivos que la acompañan.

## Uso

| Acción | Atajo |
|---|---|
| Grabar y transcribir | `⌥ Space` en Mac, `Ctrl + Space` en Windows |
| Cancelar | `Esc`, funciona durante la grabación y durante la transcripción |

Con **Mantener Presionado** activado en Ajustes, el atajo cambia: sostienes
para grabar y sueltas para transcribir.

## Cómo funciona

```
micrófono  ->  VAD  ->  Whisper  ->  diccionario  ->  portapapeles  ->  pegado
```

**VAD (detección de voz).** Antes de transcribir, [Silero](https://github.com/snakers4/silero-vad)
descarta los tramos sin voz. No es un adorno: Whisper inventa frases cuando le
das silencio ("Gracias por ver el video" es la clásica). Filtrar antes elimina
ese problema de raíz. Si no encuentra voz, la app lo dice en pantalla en vez de
cerrarse sin más.

**Modelo.** `whisper-large-v3-turbo` sobre GPU Metal en Apple Silicon, con
respaldo en CPU. Tras unos minutos sin dictar el modelo se descarga de memoria
y libera alrededor de 1.6 GB; el siguiente dictado lo recarga solo.

**Diccionario.** Dos capas sobre `dictionary.json`:

- `hints` sesga al modelo hacia la terminología de la empresa y sirve de
  referencia para corregir lo que se oyó mal. Para un término nuevo, agregarlo
  aquí suele bastar.
- `corrections` son reemplazos exactos, para errores fonéticos que no se
  parecen a la palabra correcta (`eicher` por `Azure`).

La corrección aproximada une palabras partidas (`work front` a `Workfront`) y
arregla términos mal escritos, sin tocar el español legítimo: consulta un
diccionario para no convertir "apio" en "API" ni "así" en "API".

Puedes editar el diccionario sin reinstalar. Está en
`~/Library/Application Support/QStrauss Voice/` en Mac y en `%APPDATA%\QStrauss Voice\`
en Windows, junto a tus ajustes y al log. El botón de recarga en Ajustes lo
aplica sin reiniciar.

## Ajustes

Sensibilidad del filtro de voz, posición del overlay, modo de pegado, descarga
del modelo por inactividad, mantener presionado, e inicio automático con la
sesión.

La app **no guarda grabaciones ni historial**. Nada se almacena y nada sale a
internet.

## Desarrollo

```bash
./setup_mac.sh      # o setup_win.bat
./run.sh            # arranca desde código fuente
```

Tests, sin micrófono ni conexión:

```bash
python test_vad.py      # detección de voz contra 105 muestras de ruido y 17 de habla
python test_terms.py    # corrección de términos, positivos y falsos positivos
```

Compilar:

```bash
./build_mac.sh      # o build_win.bat
```

Publicar una versión: empuja un tag `v*` y CI compila las dos plataformas y
crea la release.

Al ejecutar desde código fuente, macOS atribuye el permiso de Accesibilidad al
intérprete de Python y no a la app, así que el pegado automático no funcionará.
Para probar cualquier cosa que dependa de permisos hay que compilar el `.app`.

## Licencias

El código de este repositorio es propiedad de la empresa.

El diccionario de español en `resources/dic/` viene del proyecto
[RLA-ES](https://github.com/sbosio/rla-es) y se distribuye bajo un triple
esquema disjunto: GPLv3+, LGPLv3+ o MPL 1.1+, a elección de quien lo use. Los
textos completos y el detalle están en esa misma carpeta.
