"""
src/chatterbox_explorer/domain/languages.py
============================================
Pure Python constants for the Multilingual TTS tab.

Extracted verbatim from app.py — no framework dependencies allowed.
Allowed imports: stdlib only (none needed here — pure data).
Forbidden: torch, gradio, chatterbox, psutil, huggingface_hub.
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Language options
# ──────────────────────────────────────────────────────────────────────────────
# 23 entries in "<code> - <Name>" format.
# Used as Gradio dropdown choices for the Multilingual TTS tab.

LANGUAGE_OPTIONS: list[str] = [
    "ar - Arabic",
    "da - Danish",
    "de - German",
    "el - Greek",
    "en - English",
    "es - Spanish",
    "fi - Finnish",
    "fr - French",
    "he - Hebrew",
    "hi - Hindi",
    "it - Italian",
    "ja - Japanese",
    "ko - Korean",
    "ms - Malay",
    "nl - Dutch",
    "no - Norwegian",
    "pl - Polish",
    "pt - Portuguese",
    "ru - Russian",
    "sv - Swedish",
    "sw - Swahili",
    "tr - Turkish",
    "zh - Chinese",
]


# ──────────────────────────────────────────────────────────────────────────────
# Sample texts
# ──────────────────────────────────────────────────────────────────────────────
# Keyed by the full LANGUAGE_OPTIONS entry (e.g. "fr - French").
# Source: official Chatterbox multilingual demo.

SAMPLE_TEXTS: dict[str, str] = {
    "ar - Arabic":    "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب.",
    "da - Danish":    "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal.",
    "de - German":    "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
    "el - Greek":     "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube.",
    "en - English":   "Last month, we reached a new milestone with two billion views on our YouTube channel.",
    "es - Spanish":   "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube.",
    "fi - Finnish":   "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme.",
    "fr - French":    "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube.",
    "he - Hebrew":    "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו.",
    "hi - Hindi":     "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।",
    "it - Italian":   "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube.",
    "ja - Japanese":  "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。",
    "ko - Korean":    "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다.",
    "ms - Malay":     "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami.",
    "nl - Dutch":     "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal.",
    "no - Norwegian": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår.",
    "pl - Polish":    "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube.",
    "pt - Portuguese":"No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube.",
    "ru - Russian":   "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
    "sv - Swedish":   "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal.",
    "sw - Swahili":   "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kwenye kituo chetu cha YouTube.",
    "tr - Turkish":   "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık.",
    "zh - Chinese":   "上个月，我们达到了一个新的里程碑，我们的YouTube频道观看次数达到了二十亿次。",
}


# ──────────────────────────────────────────────────────────────────────────────
# Language audio defaults
# ──────────────────────────────────────────────────────────────────────────────
# Keyed by 2-letter ISO 639-1 code (e.g. "fr").
# Values are publicly accessible FLAC files hosted by Resemble AI on GCS.
# These load automatically when the user switches the language dropdown —
# no manual upload needed.

LANGUAGE_AUDIO_DEFAULTS: dict[str, str] = {
    "ar": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
    "da": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
    "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "el": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
    "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    "fi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
    "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "he": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
    "hi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
    "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    "ja": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
    "ko": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
    "ms": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
    "nl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
    "no": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
    "pl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
    "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
    "ru": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
    "sv": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
    "sw": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
    "tr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
    "zh": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
}


# ──────────────────────────────────────────────────────────────────────────────
# Paralinguistic tags
# ──────────────────────────────────────────────────────────────────────────────
# Supported by the Turbo model only.
# Inserted into the text prompt to trigger non-verbal vocalisations.
# Rendered as clickable buttons in the Turbo tab UI.

PARA_TAGS: list[str] = [
    "[laugh]",
    "[chuckle]",
    "[cough]",
    "[sigh]",
    "[gasp]",
    "[hmm]",
    "[clears throat]",
]
