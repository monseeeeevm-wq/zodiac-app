# -*- coding: utf-8 -*-
"""
questions.py
------------
Banco de preguntas del cuestionario de perfil de personalidad / signo zodiacal.

METODOLOGÍA DE PESOS (parte medular del proyecto)
====================================================
Cada pregunta tiene 4 opciones de respuesta. Cada opción se investigó y se
asoció (con base en rasgos astrológicos clásicos) a UNO O VARIOS signos.

Regla aritmética de asignación de peso:
    Si una opción se asocia a "k" signos, cada uno de esos signos recibe
    un peso de 1/k (reparto uniforme), y el resto de los signos (los NO
    asociados a esa opción) reciben 0.

    Ejemplo: la opción "Pregunto a varias personas antes de decidir" se
    asocia a Géminis y Libra -> peso 0.5 a Géminis, 0.5 a Libra, 0 a los demás.

Esto garantiza que:
    1) Cada pregunta aporta EXACTAMENTE 1.0 punto en total (sin importar
       a cuántos signos se asocie la opción elegida) -> todas las preguntas
       pesan lo mismo en el resultado final (ponderación justa).
    2) El vector final de un usuario (suma de sus 18 respuestas) es
       comparable entre usuarios, porque todos parten de la misma escala.

SIGNOS: los 12 signos del zodiaco.

Además de clasificar por Signo (12 categorías), el vector se puede agregar
en dos categorizaciones astrológicas adicionales, muy usadas en astrología
real y con menos categorías (más "agrupado"/binario-friendly):

    - ELEMENTO  (4 grupos): Fuego / Tierra / Aire / Agua
    - MODALIDAD (3 grupos): Cardinal / Fijo / Mutable
        Cardinal -> signos que "inician" cada estación (líderes, iniciadores)
        Fijo     -> signos que "sostienen" la estación (constantes, tercos)
        Mutable  -> signos que "cierran" la estación (adaptables, flexibles)

Cada signo aparece asociado a respuestas en varias preguntas distintas,
todos superan por mucho el mínimo de 4 preguntas requerido por el profesor
(ver coverage_report() al final del archivo).
"""

SIGNOS = [
    "Aries", "Tauro", "Géminis", "Cáncer", "Leo", "Virgo",
    "Libra", "Escorpio", "Sagitario", "Capricornio", "Acuario", "Piscis",
]

ELEMENTO_POR_SIGNO = {
    "Aries": "Fuego", "Leo": "Fuego", "Sagitario": "Fuego",
    "Tauro": "Tierra", "Virgo": "Tierra", "Capricornio": "Tierra",
    "Géminis": "Aire", "Libra": "Aire", "Acuario": "Aire",
    "Cáncer": "Agua", "Escorpio": "Agua", "Piscis": "Agua",
}
ELEMENTOS = ["Fuego", "Tierra", "Aire", "Agua"]

MODALIDAD_POR_SIGNO = {
    "Aries": "Cardinal", "Cáncer": "Cardinal", "Libra": "Cardinal", "Capricornio": "Cardinal",
    "Tauro": "Fijo", "Leo": "Fijo", "Escorpio": "Fijo", "Acuario": "Fijo",
    "Géminis": "Mutable", "Virgo": "Mutable", "Sagitario": "Mutable", "Piscis": "Mutable",
}
MODALIDADES = ["Cardinal", "Fijo", "Mutable"]

# Explicación breve de qué significa cada elemento y modalidad — se muestra
# junto al resultado para que la persona entienda qué le está diciendo
# su clasificación, no solo la etiqueta.
DESCRIPCION_ELEMENTO = {
    "Fuego": "acción, impulso y pasión. Te mueves por instinto, energía y ganas de hacer las cosas YA.",
    "Tierra": "practicidad y estabilidad. Te conectas con lo concreto, lo que se puede tocar, planear y construir.",
    "Aire": "ideas y comunicación. Vives en tu cabeza, conectas conceptos y necesitas hablar/socializar para procesar.",
    "Agua": "emoción e intuición. Sientes todo intensamente y te mueves más por lo que percibes que por la lógica.",
}

DESCRIPCION_MODALIDAD = {
    "Cardinal": "eres quien inicia. Te avientas primero, arrancas proyectos, tomas la iniciativa antes que nadie.",
    "Fijo": "eres quien sostiene. Una vez que decides algo, no te mueves fácil — constancia y lealtad son tu marca.",
    "Mutable": "eres quien se adapta. Fluyes con el cambio, te ajustas rápido y no te aferras a un solo plan.",
}

# Calibración de sesgo (Monte Carlo, ver docs/calibracion.md o el historial del chat).
# Dato curioso / "en tendencia" por signo, se muestra al final del cuestionario
DATO_CURIOSO_POR_SIGNO = {
    "Aries": "Eres el signo que siempre manda el mensaje de \"yo invito\" sin pensarlo dos veces. Literal el main character energy del zodiaco.",
    "Tauro": "Tienes fama de ser el signo más leal a su comfort food y a su gente — si Tauro te adoptó como amigo(a), es para toda la vida.",
    "Géminis": "Eres de los signos con más personalidades dentro de un chat grupal: en la mañana mandas memes, en la tarde das terapia gratis.",
    "Cáncer": "Tu superpoder es acordarte hasta del cumpleaños del perro de tu amigo. La gente confía en ti sin saber muy bien por qué.",
    "Leo": "Naciste para la cámara: si hay una foto grupal, tú ya sabes exactamente en qué ángulo te ves mejor.",
    "Virgo": "Eres el que revisa la ortografía del grupo antes de mandar el mensaje importante. Tu \"detallito\" salva proyectos enteros.",
    "Libra": "No puedes decidir ni qué pedir en un restaurante, pero eres el que mantiene la paz cuando el chat se pone tenso.",
    "Escorpio": "Tienes fama de \"intenso(a)\", pero en realidad solo odias las conversaciones superficiales. O todo o nada.",
    "Sagitario": "Ya tienes el próximo viaje planeado en tu cabeza aunque ni siquiera hayas terminado el actual. Libertad ante todo.",
    "Capricornio": "Tienes un plan a 5 años mientras el resto del grupo sigue decidiendo qué cenar hoy. Ambición nivel jefe.",
    "Acuario": "Eres el amigo random que manda un dato curiosísimo a las 2am y cambia la conversación por completo.",
    "Piscis": "Sientes las películas, las canciones y hasta los memes tristes más fuerte que nadie. Tu empatía es tu superpoder.",
}


def _pesos(signos_asociados):
    """Reparte 1.0 uniformemente entre los signos asociados a una opción."""
    if not signos_asociados:
        return {s: 0.0 for s in SIGNOS}
    w = round(1.0 / len(signos_asociados), 6)
    return {s: (w if s in signos_asociados else 0.0) for s in SIGNOS}


# Cada pregunta: id, texto, y 4 opciones (texto, signos asociados)
_RAW_QUESTIONS = [
    ("q1", "Te ofrecen dos oportunidades increíbles el mismo día. ¿Qué haces?", [
        ("Acepto la que se sienta más emocionante, ya luego resuelvo", ["Capricornio", "Sagitario"]),
        ("Hago una comparación con pros y contras de cada una", ["Virgo", "Capricornio"]),
        ("Le pregunto a mi grupo de chat qué opinan ellos", ["Géminis", "Libra"]),
        ("Voy con la que se siente bien, aunque no sepa explicar por qué", ["Cáncer", "Piscis"]),
    ]),
    ("q2", "Es viernes en la noche, ¿qué plan te prende más?", [
        ("Algo de adrenalina o un reto físico de último momento", ["Acuario", "Sagitario"]),
        ("Pijama, comida a domicilio y una serie nueva", ["Tauro", "Cáncer"]),
        ("Antro o fiesta, conocer gente nueva toda la noche", ["Acuario", "Libra"]),
        ("Algo random y misterioso, o de plano cancelar todo", ["Escorpio", "Piscis"]),
    ]),
    ("q3", "En un trabajo en equipo (escuela o chamba), tú eres quien...", [
        ("Se avienta a liderar y reparte las tareas", ["Aries", "Leo"]),
        ("Revisa que no haya ni una falta de ortografía", ["Sagitario", "Capricornio"]),
        ("Calma los dramas cuando alguien se pelea", ["Libra", "Tauro"]),
        ("Propone la idea que a nadie más se le ocurrió", ["Leo", "Géminis"]),
    ]),
    ("q4", "Alguien comenta algo feo en tus redes. ¿Qué haces?", [
        ("Respondo al toque, no me quedo callado(a)", ["Capricornio", "Virgo"]),
        ("Me lo creo y me arruina el día por completo", ["Cáncer", "Piscis"]),
        ("Analizo fríamente si tiene algo de razón", ["Tauro", "Capricornio"]),
        ("Bloqueo y sigo con mi vida, ni tiempo pierdo", ["Tauro", "Sagitario"]),
    ]),
    ("q5", "Te cae dinero extra (beca, regalo, propina grande). ¿Qué haces?", [
        ("Me lo gasto en una experiencia o viaje ya mismo", ["Aries", "Leo"]),
        ("Lo guardo, uno nunca sabe cuándo lo va a necesitar", ["Virgo", "Capricornio"]),
        ("Invito a mi gente favorita, se disfruta mejor así", ["Cáncer", "Piscis"]),
        ("Lo invierto en algo (curso, gadget, cripto, lo que sea)", ["Géminis", "Leo"]),
    ]),
    ("q6", "Truena una discusión fuerte con tu crush, pareja o mejor amigo(a). Tú...", [
        ("Digo todo lo que siento ahí mismo, sin filtro", ["Aries", "Escorpio"]),
        ("Me quedo en shock y lo proceso horas después, solo(a)", ["Tauro", "Escorpio"]),
        ("Trato de resolverlo hablando con calma", ["Géminis", "Libra"]),
        ("Lo digo derechito, sin rodeos ni drama extra", ["Acuario", "Sagitario"]),
    ]),
    ("q7", "Tu cuarto (o tu escritorio, o tu carro) se ve...", [
        ("Un caos organizado, todo en movimiento constante", ["Aries", "Géminis"]),
        ("Súper acogedor, lleno de fotos y cosas con historia", ["Tauro", "Cáncer"]),
        ("Impecable, cada cosa tiene su lugar exacto", ["Virgo", "Capricornio"]),
        ("Con una vibra única que nadie más tiene", ["Leo", "Acuario"]),
    ]),
    ("q8", "Planeas un viaje con amigos. Tú eres quien...", [
        ("Dice \"ya nos organizamos allá\", cero plan armado", ["Libra", "Virgo"]),
        ("Tiene el itinerario en Excel desde hace un mes", ["Virgo", "Capricornio"]),
        ("Solo quiere ir a donde vaya su persona favorita", ["Cáncer", "Libra"]),
        ("Busca el lugar más random que nadie más conoce", ["Escorpio", "Leo"]),
    ]),
    ("q9", "Así muestras que alguien te importa de verdad...", [
        ("Con acciones, apareces justo cuando te necesitan", ["Acuario", "Cáncer"]),
        ("Ayudándole con cosas prácticas: mudanza, tarea, lo que sea", ["Tauro", "Virgo"]),
        ("Mandando memes random y hablando horas", ["Géminis", "Libra"]),
        ("Con toda la intensidad, sin medias tintas", ["Escorpio", "Piscis"]),
    ]),
    ("q10", "Cancelan tu plan de última hora. ¿Cómo reaccionas?", [
        ("Mejor, invento algo nuevo al toque", ["Libra", "Leo"]),
        ("Me estresa, ya tenía todo listo en mi cabeza", ["Tauro", "Aries"]),
        ("Me pega más de lo que debería, ni sé por qué", ["Cáncer", "Piscis"]),
        ("Lo tomo como señal de que algo mejor viene", ["Virgo", "Leo"]),
    ]),
    ("q11", "En el chat grupal, tú eres el/la que...", [
        ("Manda mensajes directos, sin rodeos", ["Capricornio", "Sagitario"]),
        ("Pregunta \"¿todo bien?\" en cuanto alguien anda raro", ["Libra", "Tauro"]),
        ("Manda quince mensajes seguidos sin puntuación", ["Géminis", "Libra"]),
        ("Solo escribe cuando de verdad importa", ["Escorpio", "Piscis"]),
    ]),
    ("q12", "Lo que jamás perdonas en una amistad es...", [
        ("Que me dejen plantado(a) sin avisar", ["Capricornio", "Sagitario"]),
        ("Que prometan algo y no lo cumplan", ["Acuario", "Aries"]),
        ("Que no me escuchen de verdad, solo esperen su turno de hablar", ["Escorpio", "Piscis"]),
        ("Que sean aburridos, necesito buena plática", ["Géminis", "Acuario"]),
    ]),
    ("q13", "Semana de exámenes o entregas. ¿Cómo sobrevives?", [
        ("Con ejercicio, necesito gastar toda esa energía", ["Aries", "Acuario"]),
        ("Comiendo algo rico sin ninguna culpa", ["Tauro", "Escorpio"]),
        ("Desaparezco del radar un rato, nadie me encuentra", ["Escorpio", "Piscis"]),
        ("Haciendo listas y horarios hasta el último minuto", ["Virgo", "Aries"]),
    ]),
    ("q14", "Vas a aprender algo nuevo (skill, idioma, deporte). Tú...", [
        ("Te avientas sin tutorial, aprendes cayéndote", ["Aries", "Escorpio"]),
        ("Practicas todos los días hasta dominarlo por completo", ["Virgo", "Capricornio"]),
        ("Ves veinte videos de YouTube antes de siquiera empezar", ["Géminis", "Virgo"]),
        ("Te armas tu propio método, paso a paso", ["Sagitario", "Acuario"]),
    ]),
    ("q15", "En una fiesta o reunión, tú eres...", [
        ("El alma de la fiesta, la pasas increíble", ["Acuario", "Sagitario"]),
        ("Quien conoce a todos y platica con todos", ["Géminis", "Libra"]),
        ("La que se queda en su grupito chiquito de siempre", ["Cáncer", "Tauro"]),
        ("La que observa todo desde una esquina, con su propia vibra", ["Escorpio", "Acuario"]),
    ]),
    ("q16", "Lo que de verdad te mueve en la vida es...", [
        ("Ser el/la mejor en lo que hago", ["Aries", "Leo"]),
        ("Tener estabilidad y no batallar tanto", ["Tauro", "Cáncer"]),
        ("Conectar de verdad con la gente que amo", ["Piscis", "Libra"]),
        ("Entender cómo funciona absolutamente todo", ["Géminis", "Sagitario", "Leo"]),
    ]),
    ("q17", "Metiste la pata en algo importante. ¿Qué haces?", [
        ("Sigo adelante, ya fue, no me quedo pensando en eso", ["Aries", "Virgo"]),
        ("Me atormento días enteros dándole vueltas", ["Sagitario", "Aries"]),
        ("Me destroza emocionalmente por un buen rato", ["Cáncer", "Piscis"]),
        ("\"Ya qué, mañana es otro día\"", ["Sagitario", "Géminis", "Leo"]),
    ]),
    ("q18", "Tu rutina ideal de vida sería...", [
        ("Que cada día sea distinto, odio la monotonía", ["Capricornio", "Géminis", "Leo"]),
        ("Bien estructurada, saber exactamente qué esperar", ["Escorpio", "Sagitario", "Capricornio"]),
        ("Girando alrededor de mi familia y mi gente cercana", ["Cáncer", "Piscis"]),
        ("Con espacio para brillar y hacer mi propia cosa", ["Leo", "Acuario"]),
    ]),
]

QUESTIONS = []
for qid, text, options in _RAW_QUESTIONS:
    opts = []
    for opt_text, signos in options:
        opts.append({
            "text": opt_text,
            "signos": signos,
            "weights": _pesos(signos),
        })
    QUESTIONS.append({"id": qid, "text": text, "options": opts})


def coverage_report():
    """Cuenta en cuántas preguntas (>0 peso) aparece cada signo. Debug/QA."""
    cov = {s: 0 for s in SIGNOS}
    for q in QUESTIONS:
        signos_en_pregunta = set()
        for opt in q["options"]:
            for s in opt["signos"]:
                signos_en_pregunta.add(s)
        for s in signos_en_pregunta:
            cov[s] += 1
    return cov


if __name__ == "__main__":
    print("Cobertura de preguntas por signo (mínimo requerido: 4):")
    for s, c in coverage_report().items():
        estado = "OK" if c >= 4 else "FALTA"
        print(f"  {s:12s}: {c} preguntas  [{estado}]")
    print(f"\nTotal de preguntas: {len(QUESTIONS)}")
    print(f"Total de signos: {len(SIGNOS)}")
