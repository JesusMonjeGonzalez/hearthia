# Hearthia como agente de uso diario: evaluación y primer refuerzo

**Actualización posterior:** se han aplicado sesiones SQLite, workers de
herramientas cancelables, workspace/AGENTS.md, paginación y presupuesto de contexto
por modelo. El estado de esos bloques y sus límites está en
[`CHAT-HARNESS.md`](CHAT-HARNESS.md). La evaluación siguiente conserva el diagnóstico
de partida; varias carencias de las fases 1–3 ya tienen una primera implementación.
El bloque posterior añade edición, comandos y verificación en modo Develop:
[`CODING-AGENT.md`](CODING-AGENT.md), con una prueba de navegador que ejecuta
realmente el ciclo de fallo, corrección y pruebas sobre un proyecto temporal.

## Veredicto

Hearthia tiene una base funcional de **gestión de inferencia local**, pero todavía
no es un sustituto de Pi o Crush para trabajar con repositorios y el equipo.
Su ventaja potencial es combinar un agente ligero con la gestión explícita de
la memoria unificada del Mac. Conviene conservar esa ventaja y construir un
harness pequeño, observable y cancelable sobre ella.

Esta revisión se basa en el código de `api/chat.py`, `api/tools.py`, `gateway.py`,
`web/chat.js`, `api/repomap.py`, `brain/search.py`, los caminos de carga y sus
pruebas. La suite inicial fue de 559 pruebas aprobadas. No constituye una
certificación de estabilidad con modelos reales ni un benchmark de RAM.
Había modificaciones previas en el árbol de trabajo; se han preservado.

## Comparación funcional

| Área | Evidencia en Hearthia | Distancia hasta un agente competitivo |
|---|---|---|
| Inferencia local | Gestión de llama.cpp/llama-swap, estimaciones GGUF, TTL, calibración y telemetría | Es la base más diferenciada; falta demostrar comportamiento bajo presión real y cubrir todas las entradas de inferencia |
| Conversación | Chat web con streaming, Stop, historial y exportación | Faltan sesiones durables independientes del navegador, recuperación de tareas, seguimiento de herramientas y edición/reintento de turnos |
| Herramientas | Lectura, búsqueda, listado, glob y búsqueda de notas | No hay edición ni shell en el chat; no puede completar el ciclo editar → ejecutar → corregir → verificar |
| Harness | Bucle de hasta ocho rondas y deduplicación durante un turno | Falta separación entre ejecución, sesión, eventos y UI; no conserva el rastro completo de herramientas entre turnos |
| Contexto | Mapas de proyecto y recorte de resultados antiguos | Falta presupuesto por tokenizer/contexto efectivo del modelo, reserva de salida y compactación reanudable |
| Extensiones | Servidor MCP y fachada TreePact | Un servidor MCP expone Hearthia a otros agentes; no equivale a un cliente MCP que permita al chat usar otras herramientas |
| Equipo y proyectos | Acceso de lectura con los permisos del proceso | Falta workspace explícito por sesión, instrucciones de proyecto, ejecución de procesos cancelable y conectores seleccionables |
| Calidad observada | Suite de Python, estáticos y pruebas de navegador opcionales | No hay evidencia suficiente de tareas de programación end-to-end con modelos reales ni de una sesión larga estable |

Crush documenta sesiones por proyecto, cambio de modelo, LSP, cliente MCP,
permisos y skills. Pi se presenta como un agente mínimo extensible con CLI,
modos JSON/RPC y SDK. Eso marca el objetivo funcional; no demuestra que alguno
consuma menos RAM en este Mac. No se han ejecutado comparativas entre procesos.

Fuentes consultadas el 28-09-2026:
- https://github.com/charmbracelet/crush/blob/main/README.md
- https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md
  (actualmente remite al proyecto `earendil-works/pi`).

## Problemas encontrados y cambios de esta revisión

1. **El chat evitaba el control de admisión de memoria.** Ahora comprueba el
   inventario y el presupuesto antes de cada ronda. Inventario desconocido o
   modelo desconocido se rechazan en modo `enforce`. Búsqueda de notas comprueba
   también el modelo de embeddings antes de usarlo.
2. **Trabajo concurrente sin límite en el chat.** Un solo turno activo por
   instancia de daemon; otro turno recibe rechazo, sin cola creciente. Esto no
   constituye una reserva global frente a CLI, MCP, warm o clientes directos.
3. **Indexación implícita cara.** Mencionar una carpeta ya no construye
   automáticamente su índice vectorial ni carga embeddings. Se mantienen los
   mapas de proyecto y herramientas de lectura. Las utilidades de indexado
   semántico permanecen disponibles en el código para trabajo posterior.
4. **Lecturas aparentemente pequeñas con coste grande.** Lectura de archivos y
   previews ahora leen solo un prefijo. Lotes de lectura limitados a 24 archivos;
   glob usa recorrido podado con profundidad y número de archivos limitados;
   caché de mapas limitada a 16 raíces. No es un sandbox ni limita toda operación
   posible de filesystem.
5. **Entrada y salida poco robustas.** Petición máxima de 1 MB, validación de
   mensajes y sampling, hasta 16 llamadas de herramienta por ronda, stream
   acotado y errores explícitos. El contexto serializado tiene un techo de
   60.000 caracteres tras el recorte; si no cabe se rechaza, sin borrar mensajes
   del usuario. No es aún una medición de tokens ni compactación automática.
6. **Streaming frágil.** Decodificación UTF-8 incremental, un marcador final
   `[DONE]` por turno normal y comprobación del estado HTTP upstream. Cancelación
   libera el turno y cierra el generador upstream, cubierto por regresión.
7. **Uso de herramientas y prompt.** Se evita ejecutar duplicados dentro de la
   misma ronda y se devuelven errores de argumentos como resultados. El prompt
   deja de anunciar escritura inexistente. Los extractos recuperados dejan de
   presentarse como una respuesta necesariamente correcta.
8. **Coste del navegador.** Durante el streaming se renderiza texto escapado;
   Markdown y resaltado se aplican al terminar. No se fuerza scroll cuando el
   usuario está leyendo arriba. Fallos al guardar no bloquean la liberación del
   estado de envío. El contador aproximado se etiqueta como chunks/s, no tokens/s;
   las métricas reales del backend conservan su etiqueta de tokens.

## Lo que aún puede comprometer memoria o fluidez

- El puerto directo de llama-swap sigue admitiendo clientes sin pasar por este
  chat. El control de admisión no es todavía una transacción compartida por
  todas las superficies. Dos entradas diferentes pueden competir por RAM.
- El modo `warn`/`off`, las estimaciones de cabeceras desconocidas y la actividad
  de otras aplicaciones impiden prometer «nunca saturará el Mac».
- El contexto de caracteres no se adapta al tokenizer ni a los slots/contexto
  efectivos de llama.cpp; modelos pequeños pueden rechazar prompts que pasan
  este límite. Falta tokenización o una contabilidad conservadora por modelo.
- Herramientas de filesystem siguen ejecutándose en el event loop. Una regex
  costosa o un volumen lento puede bloquearlo. Hace falta un trabajador aislado
  con plazo y terminación real; un timeout sobre un thread no basta.
- Historial de navegador, adjuntos y DOM de conversaciones no están paginados.
  La cuota de localStorage puede agotarse; el aviso no reemplaza una migración
  a almacenamiento durable y carga bajo demanda.
- Caches de índices semánticos usadas por otras rutas necesitan eviction,
  invalidación y cierre de SQLite. El chat normal evita ahora activar ese camino.

## Orden recomendado para llegar al reemplazo

### 1. Runtime estable y medible

Un servicio único de admisión/reserva para chat, warm, loadouts y herramientas;
presión de memoria/swap observada durante ejecución; presupuesto de contexto por
modelo; herramientas en trabajadores cancelables. Probar carreras entre rutas.

Aceptación: dos solicitudes de carga competidoras no sobreasignan; detener un
turno libera la conexión y sus trabajadores; una sesión de 30 minutos no muestra
crecimiento sostenido del heap del daemon o del navegador tras repetición de la
misma tarea. Registrar hardware, modelos, contextos y otras cargas del Mac.

### 2. Agente capaz de terminar tareas

Sesiones SQLite con eventos incrementales y trazas de herramientas; workspace
por sesión; instrucciones AGENTS.md; lectura por rangos; edición mediante patch;
shell con cwd, salida acotada, plazo y terminación de grupo de procesos. Mostrar
diffs y resultado de verificaciones. Definir con el usuario el alcance de las
acciones de escritura/ejecución antes de habilitarlas.

Aceptación: desde Hearthia, abrir un repo de prueba, localizar un bug, corregirlo,
ejecutar sus pruebas y presentar el diff. Reanudar tras reiniciar el daemon sin
perder qué se ejecutó ni repetir efectos de herramientas por accidente.

### 3. Experiencia de conversación comparable

Separar mensajes, actividad de herramientas y estado del turno; añadir
reintento/edición, cola breve de seguimiento, contexto/uso visible, búsqueda de
sesiones, exportación completa y carga paginada. Mantener una UI ligera antes de
considerar una TUI adicional.

### 4. Interacción ampliada con el equipo

Cliente MCP y conectores elegidos para las aplicaciones que realmente use el
usuario; perfiles de agente como instrucciones/herramientas sobre un mismo
runtime. Evitar un modelo residente por agente. Introducir concurrencia de
agentes solo con un presupuesto global y mediciones que justifiquen su coste.

## Verificación reproducible

Desde la raíz del repositorio:

```sh
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
node --input-type=module --check < src/hearthia/web/chat.js
```

Las pruebas nuevas simulan gateway y presión/admisión: no cargan modelos reales.
El test de cancelación verifica cierre de la conexión/generador, no la latencia
con que una versión concreta de llama.cpp detiene su trabajo GPU.

Resultado al cerrar la revisión: **580 pruebas aprobadas**, Ruff lint/formato,
mypy (47 archivos) y sintaxis JavaScript correctos. No se ejecutó un navegador
real ni una carga de inferencia/benchmark de memoria; esas validaciones siguen
pendientes antes de afirmar que está listo para sustituir al agente diario.
