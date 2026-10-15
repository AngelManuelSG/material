"""Bot de El Guardián: ensemble de señales por oración (LLM juez + NLI + comprobaciones de texto).

Convención del kit/organizador: score = probabilidad de que la respuesta ALUCINE.
score >= 0.5 -> alucina. En ejecución solo usa la librería estándar; los pesos se
recalculan con `calibrar_guardian.py` (numpy) cuando cambie el train.

Cómo decide:
  1. Cada oración de la respuesta se convierte en señales en [0,1] orientadas a
     "alucina" (ver `senales_oracion`).
  2. z_oracion = SESGO + Σ PESOS[s] · señal_s          (regresión logística)
  3. z_caso = max(z_oracion): basta UNA afirmación falsa.
  4. score = sigmoide(z_caso - CORTE): CORTE es el z que minimiza 4·FN + FP en la
     validación, así el corte del organizador (0.5) cae en el sitio óptimo.

Patrones estadísticos que justifican los pesos (ver `calibrar_guardian.py`):
  · LLM juez: si rechaza una oración acierta >90 % (precisión alta) -> pesos grandes.
    Si la aprueba, se le escapan sobre todo distorsiones y sobregeneralizaciones,
    y su recall baja cuantas más fuentes hay -> la duda (1 - confianza) pesa más
    con muchas fuentes (`duda_x_fuentes`).
  · NLI: fiable para lo que NINGUNA fuente respalda (no respaldado, distorsión);
    su "contradicción" y la similitud dan falsas alarmas con varias fuentes, por eso
    no se usan (peso 0 en la calibración).
  · Texto: causas añadidas (no respaldado), matices perdidos "más de / solo /
    podría / prevé" (sobregeneralización, distorsión). Las cifras se descartaron:
    las respuestas fieles hacen aritmética y el signo de la señal cambiaba.
  · Gravedad de los errores: Contradicción > Distorsión > Sobregeneralización =
    No respaldado. Se aplica al calibrar (GRAVEDAD_TIPO en calibrar_guardian.py).
"""
import math
import re
import unicodedata

# ======================= PESOS (editables) =======================
# Salen de `python calibrar_guardian.py --datos train.jsonl`. Un peso mayor = esa
# señal empuja más hacia "alucina". 0 = señal desactivada.
SESGO = -1.57
PESOS = {
    "llm_contradicha": 3.42,   # LLM dice "contradicha" (· confianza). Caza contradicciones.
    "llm_no_respaldada": 2.28, # LLM dice "no_respaldada" (· confianza). Caza no respaldado.
    "llm_nulo": 0.56,          # el LLM no contestó: sin juez, algo de sospecha.
    "llm_duda": 1.78,          # LLM dice "respaldada" pero con poca confianza (1 - conf).
    "duda_x_fuentes": 1.61,    # la misma duda, escalada por nº de fuentes: más fuentes = LLM menos fiable.
    "nli_mejor_fuente": 0.69,  # 1 - respaldo de la fuente que más respalda la oración.
    "nli_fragmento": 0.81,     # peor fragmento de la oración frente a su mejor fuente.
    "nli_todas": 0.00,         # 1 - respaldo_todas (redundante con las dos anteriores).
    "nli_contradiccion": 0.00, # contradicción NLI: falsas alarmas con varias fuentes.
    "causa_nueva": 0.40,       # "gracias a / debido a / porque..." que ninguna fuente dice.
    "matiz_perdido": 0.79,     # la fuente citada dice "más de / solo / algunos..." y la oración no.
    "condicional_perdido": 1.64,  # la fuente citada dice "podría / prevé / propone..." y la oración lo afirma.
    "universal_nuevo": 0.00,   # "todos / siempre / del país..." que no está en las fuentes.
}
CORTE = -1.41      # z que deja el corte óptimo (4·FN + FP) en score 0.5. Más bajo = marca MÁS.
FUENTES_REF = 10   # nº de fuentes con el que `duda_x_fuentes` llega a su máximo.
SIN_ORACIONES = 0.6  # score si el caso no trae oraciones (ante la duda, alucina: FN cuesta 4).

# =============== Patrones de texto (genéricos, del criterio de EJEMPLOS.md) ===============
CAUSA = r"\b(porque|debido a|gracias a|a causa de|por lo que|ya que|lo que provoco|lo que permitio|como consecuencia|con el fin de|motivo|razon|provoco|permitio|impulsad[oa] por)\b"
MATIZ = r"\b(mas de|menos de|alrededor de|cerca de|unos|unas|aproximadamente|casi|hasta|al menos|en torno a|la mayoria|algunos|algunas|varios|varias|parte de|solo|salvo|excepto|si se|siempre que)\b"
CONDICIONAL = r"\b(podria|podrian|puede|pueden|preve|prevee|previsto|prevista|estudia|estudian|propone|propuso|propuesta|planea|plantea|espera|esperan|estima|estiman|proyecto|posible|posiblemente|probable|objetivo|pretende|intenta|negocia|negociacion|pendiente|borrador)\b"
UNIVERSAL = r"\b(todos|todas|todo el|toda la|siempre|nunca|ningun|ninguna|cualquier|totalidad|en general|generalmente|del pais|nacional|completamente|exclusivamente)\b"


def _norm(texto):
    return unicodedata.normalize("NFKD", (texto or "").lower()).encode("ascii", "ignore").decode()


def _maximo(valores, defecto=0.0):
    return max(valores) if valores else defecto


def senales_oracion(caso):
    """Una lista (una entrada por oración) de dicts {señal: valor en [0,1]}."""
    s = caso.get("senales") or {}
    fuentes = [_norm(f) for f in caso.get("fuentes") or []]
    todas = " ".join(fuentes)
    factor_fuentes = min(1.0, math.log(max(len(fuentes), 1)) / math.log(FUENTES_REF))
    fragmentos = s.get("fragmentos") or []
    salida = []
    for i, o in enumerate(s.get("oraciones") or []):
        llm = o.get("llm") or {}
        veredicto = llm.get("veredicto")
        conf = float(llm.get("confianza") or 0.0)
        k = llm.get("fuente")
        citada = fuentes[k] if isinstance(k, int) and 0 <= k < len(fuentes) else todas
        texto = _norm(o.get("texto"))
        frs = [f for f in fragmentos if f.get("oracion") == i] or [o]
        duda = 1.0 - conf if veredicto == "respaldada" else 0.0
        matices_fuente = set(re.findall(MATIZ, citada))
        salida.append({
            "llm_contradicha": conf if veredicto == "contradicha" else 0.0,
            "llm_no_respaldada": conf if veredicto == "no_respaldada" else 0.0,
            "llm_nulo": 0.0 if veredicto else 1.0,
            "llm_duda": duda,
            "duda_x_fuentes": duda * factor_fuentes,
            "nli_mejor_fuente": 1.0 - _maximo(o.get("respaldo_por_fuente") or []),
            "nli_fragmento": _maximo([1.0 - _maximo(f.get("respaldo_por_fuente") or []) for f in frs]),
            "nli_todas": 1.0 - float(o.get("respaldo_todas") or 0.0),
            "nli_contradiccion": max(float(o.get("contradiccion_todas") or 0.0),
                                     _maximo(o.get("contradiccion_por_fuente") or [])),
            "causa_nueva": 1.0 if any(m not in todas for m in re.findall(CAUSA, texto)) else 0.0,
            "matiz_perdido": 1.0 if matices_fuente and not re.search(MATIZ, texto) else 0.0,
            "condicional_perdido": 1.0 if re.search(CONDICIONAL, citada) and not re.search(CONDICIONAL, texto) else 0.0,
            "universal_nuevo": 1.0 if any(m not in todas for m in re.findall(UNIVERSAL, texto)) else 0.0,
        })
    return salida


class Guardian:
    def score(self, caso):
        oraciones = senales_oracion(caso)
        if not oraciones:
            return SIN_ORACIONES
        z = max(SESGO + sum(PESOS[n] * v for n, v in f.items()) for f in oraciones)
        return 1.0 / (1.0 + math.exp(-(z - CORTE)))
