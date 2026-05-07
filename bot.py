"""
Telegram Team Assistant Bot
============================
Haystack 2.x + Pinecone + OpenAI (via proxyapi.ru) + Docling + Whisper audio transcription

Audio pipeline (Haystack RemoteWhisperTranscriber):
  voice/audio message → download OGG → convert to MP3 (pydub) →
  RemoteWhisperTranscriber (Whisper-1 via proxyapi.ru) →
  transcribed text → index to Pinecone + session buffer
"""

import os
import logging
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import telebot
from telebot.types import Message
from dotenv import load_dotenv

# Load .env early so FFMPEG_PATH and other vars are available for imports below
load_dotenv()

# ── Haystack 2.x imports ─────────────────────────────────────────────────────
from haystack import Document, Pipeline
from haystack.components.audio import RemoteWhisperTranscriber
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

# ── Audio conversion (OGG Telegram → MP3 for Whisper) ────────────────────────
# Add ffmpeg directory to PATH before importing pydub so it finds the binary on import
_ffmpeg_exe = os.environ.get("FFMPEG_PATH", "")
if _ffmpeg_exe:
    _ffmpeg_dir = str(Path(_ffmpeg_exe).parent)
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

try:
    from pydub import AudioSegment

    # Also set explicitly on the class as a fallback
    if _ffmpeg_exe:
        AudioSegment.converter = _ffmpeg_exe
        AudioSegment.ffprobe   = str(Path(_ffmpeg_exe).parent / "ffprobe.exe")

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not installed. Audio will be sent as-is (OGG may fail on some endpoints).")

# ── Docling ───────────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not installed. Document processing will be limited.")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment variables ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
PINECONE_API_KEY     = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME  = os.environ.get("PINECONE_INDEX_NAME", "team-assistant")
OPENAI_BASE_URL      = os.environ.get("OPENAI_BASE_URL", "https://openai.api.proxyapi.ru/v1")
# Support both PROXYAPI_API_KEY (per requirements) and legacy PROXY_API_KEY / OPENAI_API_KEY
OPENAI_API_KEY       = (
    os.environ.get("PROXYAPI_API_KEY")
    or os.environ.get("PROXY_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
if not OPENAI_API_KEY:
    raise RuntimeError("Set PROXYAPI_API_KEY (or PROXY_API_KEY / OPENAI_API_KEY) in .env")
OPENAI_MODEL         = os.environ.get("OPENAI_MODEL") or os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
WHISPER_MODEL        = os.environ.get("WHISPER_MODEL", "whisper-1")

# text-embedding-3-small → 1536 dims; text-embedding-3-large → 3072
EMBEDDING_DIM = 1536 if "small" in EMBEDDING_MODEL else 3072

# ── OpenAI via proxyapi custom API base ───────────────────────────────────────
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL


# ══════════════════════════════════════════════════════════════════════════════
#  Pinecone document store (Haystack 2.x)
# ══════════════════════════════════════════════════════════════════════════════

def build_document_store() -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=Secret.from_token(PINECONE_API_KEY),
        index=PINECONE_INDEX_NAME,
        namespace="chat-history",
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )


document_store = build_document_store()


# ══════════════════════════════════════════════════════════════════════════════
#  Haystack Pipelines
# ══════════════════════════════════════════════════════════════════════════════

def build_indexing_pipeline() -> Pipeline:
    """Embed documents and write them to Pinecone."""
    pipe = Pipeline()
    pipe.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=EMBEDDING_MODEL,
            api_base_url=OPENAI_BASE_URL,
        ),
    )
    pipe.add_component("writer", DocumentWriter(document_store=document_store))
    pipe.connect("embedder.documents", "writer.documents")
    return pipe


def build_rag_pipeline() -> Pipeline:
    """Retrieve relevant docs → build prompt → generate answer with OpenAI."""
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=8)

    prompt_template = """
Ты умный командный ассистент. На основе истории переписки ответь на вопрос.

Контекст из истории чата:
{% for doc in documents %}
---
[{{ doc.meta.get('timestamp', '') }}] {{ doc.meta.get('author_name', 'Unknown') }}:
{{ doc.content }}
{% endfor %}

Вопрос: {{ query }}

Дай подробный ответ, ссылаясь на конкретных участников и время их высказываний.
"""

    pipe = Pipeline()
    pipe.add_component(
        "text_embedder",
        OpenAITextEmbedder(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=EMBEDDING_MODEL,
            api_base_url=OPENAI_BASE_URL,
        ),
    )
    pipe.add_component("retriever", retriever)
    # Note: do NOT use required_variables here — "documents" arrives via pipeline
    # connection from retriever, not from run() input. required_variables only
    # applies to variables that must be supplied at run() time.
    pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipe.add_component(
        "generator",
        OpenAIGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
            api_base_url=OPENAI_BASE_URL,
        ),
    )
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    return pipe


def build_chat_filter(chat_id: int) -> dict:
    return {"field": "meta.chat_id", "operator": "==", "value": str(chat_id)}


def build_summary_pipeline() -> Pipeline:
    """Summarise a dialogue and produce conclusions/verdict."""
    prompt_template = """
Ты умный командный ассистент. Проанализируй следующий диалог команды.

Диалог:
{{ dialogue }}

Задачи:
1. Напиши краткое РЕЗЮМЕ обсуждения.
2. Если это был СПОР — выскажи своё аргументированное мнение, кто прав и почему.
3. Если принималось РЕШЕНИЕ — чётко сформулируй финальное решение и следующие шаги.
4. Выдели ключевые договорённости и action-items с ответственными (если названы).

Формат ответа — чёткий, структурированный, на русском языке.
"""

    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template, required_variables=["dialogue"]))
    pipe.add_component(
        "generator",
        OpenAIGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
            api_base_url=OPENAI_BASE_URL,
        ),
    )
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    return pipe


def build_doc_analysis_pipeline() -> Pipeline:
    """Analyse content of an uploaded document."""
    prompt_template = """
Ты аналитик документов. Изучи содержимое документа и выполни задание.

Содержимое документа:
{{ doc_content }}

Задание: {{ task }}

Дай подробный, структурированный ответ на русском языке.
"""
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template, required_variables=["doc_content", "task"]))
    pipe.add_component(
        "generator",
        OpenAIGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
            api_base_url=OPENAI_BASE_URL,
        ),
    )
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    return pipe


def build_transcription_pipeline() -> Pipeline:
    """Transcribe audio file → Haystack Document with text content.

    Uses Haystack's RemoteWhisperTranscriber which calls OpenAI Whisper API
    (or any OpenAI-compatible endpoint, e.g. proxyapi.ru).

    Input:  {"transcriber": {"sources": ["/path/to/audio.mp3"]}}
    Output: result["transcriber"]["documents"][0].content  →  transcribed text
    """
    pipe = Pipeline()
    pipe.add_component(
        "transcriber",
        RemoteWhisperTranscriber(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=WHISPER_MODEL,
            api_base_url=OPENAI_BASE_URL,
            # Pass language hint to improve accuracy for multilingual teams.
            # None = auto-detect, or set e.g. language="ru"
        ),
    )
    return pipe


# Initialise pipelines once
indexing_pipeline      = build_indexing_pipeline()
rag_pipeline           = build_rag_pipeline()
summary_pipeline       = build_summary_pipeline()
doc_analysis_pipeline  = build_doc_analysis_pipeline()
transcription_pipeline = build_transcription_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
#  In-memory state
# ══════════════════════════════════════════════════════════════════════════════

# chat_id → list of dicts {author_id, author_name, text, timestamp}
listening_sessions: dict[int, list[dict]] = {}

# chat_id → bool
listening_active: dict[int, bool] = {}

# Lock for thread safety (telebot uses threads)
_lock = threading.Lock()


def is_listening(chat_id: int) -> bool:
    return listening_active.get(chat_id, False)


def format_author(msg: Message) -> str:
    u = msg.from_user
    if u is None:
        return "Unknown"
    name = u.full_name or ""
    if u.username:
        name += f" (@{u.username})"
    return name.strip() or str(u.id)


# ══════════════════════════════════════════════════════════════════════════════
#  Index a single message into Pinecone
# ══════════════════════════════════════════════════════════════════════════════

def index_message(chat_id: int, author_id: int, author_name: str, text: str, timestamp: str,
                  extra_meta: Optional[dict] = None):
    meta = {
        "chat_id": str(chat_id),
        "author_id": str(author_id),
        "author_name": author_name,
        "timestamp": timestamp,
        "type": "chat_message",
    }
    if extra_meta:
        meta.update(extra_meta)
    doc = Document(content=text, meta=meta)
    try:
        indexing_pipeline.run({"embedder": {"documents": [doc]}})
        logger.info("Indexed message from %s in chat %d", author_name, chat_id)
    except Exception as e:
        logger.error("Failed to index message: %s", e)


def index_documents_batch(documents: list[Document]):
    """Index a batch of documents (used after session stop or doc upload)."""
    if not documents:
        return
    # Split long documents before embedding (word-based split avoids nltk dependency)
    splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=20)
    split_docs = splitter.run(documents=documents)["documents"]
    try:
        indexing_pipeline.run({"embedder": {"documents": split_docs}})
        logger.info("Indexed %d document chunks", len(split_docs))
    except Exception as e:
        logger.error("Failed to index documents batch: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  Audio helpers
# ══════════════════════════════════════════════════════════════════════════════

# Telegram sends voice messages as OGG/OPUS and audio files in various formats.
# Whisper API accepts: mp3, mp4, mpeg, mpga, m4a, wav, webm.
# We convert OGG → MP3 using pydub (requires ffmpeg on PATH).

AUDIO_MIME_SUPPORTED = {
    "audio/mpeg", "audio/mp4", "audio/mp3",
    "audio/wav", "audio/x-wav",
    "audio/webm", "audio/ogg",
    "audio/m4a", "audio/x-m4a",
}


def _convert_ogg_to_mp3(ogg_path: str) -> str:
    """Convert OGG/OPUS file to MP3. Returns new file path."""
    mp3_path = ogg_path.replace(".ogg", ".mp3").replace(".oga", ".mp3")
    if not mp3_path.endswith(".mp3"):
        mp3_path += ".mp3"
    if PYDUB_AVAILABLE:
        audio = AudioSegment.from_file(ogg_path, format="ogg")
        audio.export(mp3_path, format="mp3")
    else:
        # Fallback: try ffmpeg directly via subprocess
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, mp3_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return mp3_path


def transcribe_audio_file(audio_path: str) -> Optional[str]:
    """Run Haystack RemoteWhisperTranscriber on a local audio file.

    Returns the transcribed text, or None on failure.
    Whisper auto-detects language — works for Russian, English, mixed.
    """
    try:
        result = transcription_pipeline.run(
            {"transcriber": {"sources": [audio_path]}}
        )
        docs = result["transcriber"]["documents"]
        if docs and docs[0].content:
            return docs[0].content.strip()
        return None
    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return None


def _download_and_transcribe(
    msg: Message,
    file_id: str,
    duration: int,
    is_voice: bool,
    caption: str = "",
) -> None:
    """Download audio from Telegram, convert if needed, transcribe via Whisper.

    This function runs in a background thread.
    After transcription:
      - sends the transcript back to chat (with 🎙 prefix)
      - indexes it into Pinecone
      - adds to the active session buffer if listening
      - if bot is @mentioned in caption, also answers via RAG
    """
    chat_id   = msg.chat.id
    author_id = msg.from_user.id if msg.from_user else 0
    author_name = format_author(msg)
    timestamp = datetime.fromtimestamp(msg.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    username = get_bot_username()

    kind = "голосовое сообщение" if is_voice else "аудиофайл"

    # Limit: skip very long audio to avoid huge API bills (configurable)
    MAX_DURATION_SECONDS = 600  # 10 min
    if duration and duration > MAX_DURATION_SECONDS:
        bot.reply_to(
            msg,
            f"⚠️ {kind.capitalize()} слишком длинное ({duration} сек). "
            f"Максимум — {MAX_DURATION_SECONDS} сек.",
        )
        return

    bot.send_chat_action(chat_id, "typing")

    # 1. Download from Telegram
    try:
        file_info  = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
    except Exception as e:
        bot.reply_to(msg, f"❌ Не удалось скачать {kind}: {e}")
        return

    # 2. Save to temp file
    suffix = ".ogg" if is_voice else (Path(file_info.file_path).suffix or ".ogg")
    tmp_audio = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_audio.write(downloaded)
    tmp_audio.close()
    audio_path = tmp_audio.name
    converted_path: Optional[str] = None

    try:
        # 3. Convert OGG → MP3 if needed (Whisper prefers MP3)
        if audio_path.endswith(".ogg") or audio_path.endswith(".oga") or is_voice:
            try:
                converted_path = _convert_ogg_to_mp3(audio_path)
                transcribe_path = converted_path
            except Exception as conv_err:
                logger.warning("OGG→MP3 conversion failed (%s), using raw OGG", conv_err)
                transcribe_path = audio_path
        else:
            transcribe_path = audio_path

        # 4. Transcribe via Haystack RemoteWhisperTranscriber
        bot.reply_to(
            msg,
            f"🎙 Транскрибирую {kind} ({duration} сек)…",
        )
        transcript = transcribe_audio_file(transcribe_path)

        if not transcript:
            bot.reply_to(msg, f"❌ Не удалось распознать речь в {kind}.")
            return

        # 5. Post transcript to chat
        display_text = (
            f"🎙 *Транскрипция* [{author_name}]:\n_{transcript}_"
        )
        bot.send_message(chat_id, display_text)

        # 6. Index into Pinecone (background-safe — we're already in a thread)
        index_message(chat_id, author_id, author_name, transcript, timestamp,
                      extra_meta={"type": "voice_message", "duration_sec": str(duration)})

        # 7. Add to active session buffer
        if is_listening(chat_id):
            with _lock:
                if chat_id in listening_sessions:
                    listening_sessions[chat_id].append(
                        {
                            "author_id":   author_id,
                            "author_name": author_name,
                            "text":        f"[аудио] {transcript}",
                            "timestamp":   timestamp,
                        }
                    )

        # 8. If caption contains bot mention → answer via RAG
        if caption and username and re.search(rf"@{re.escape(username)}\b", caption, flags=re.IGNORECASE):
            question = re.sub(rf"@{re.escape(username)}\b", "", caption, flags=re.IGNORECASE).strip() or transcript
            try:
                result = rag_pipeline.run(
                    {
                        "text_embedder": {"text": question},
                        "retriever": {"filters": build_chat_filter(chat_id)},
                        "prompt_builder": {"query": question},
                    }
                )
                answer = result["generator"]["replies"][0]
                bot.send_message(chat_id, answer)
            except Exception as e:
                bot.send_message(chat_id, f"❌ Ошибка RAG: {e}")

    finally:
        Path(audio_path).unlink(missing_ok=True)
        if converted_path:
            Path(converted_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Docling document processing
# ══════════════════════════════════════════════════════════════════════════════

def process_document_with_docling(file_path: str) -> Optional[str]:
    """Convert document to text using Docling."""
    if not DOCLING_AVAILABLE:
        # Fallback: read raw text if possible
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error("Docling conversion error: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Bot setup
# ══════════════════════════════════════════════════════════════════════════════

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="Markdown")
BOT_USERNAME: Optional[str] = None  # filled after bot starts


def get_bot_username() -> str:
    """Return the bot's @username (without @). Cached after first successful call."""
    global BOT_USERNAME
    if not BOT_USERNAME:
        try:
            BOT_USERNAME = bot.get_me().username or ""
        except Exception as e:
            logger.error("get_bot_username() failed: %s", e)
            BOT_USERNAME = ""
    return BOT_USERNAME


# ══════════════════════════════════════════════════════════════════════════════
#  /start_listen  — begin recording session
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(commands=["start_listen"])
def cmd_start_listen(msg: Message):
    chat_id = msg.chat.id
    with _lock:
        if is_listening(chat_id):
            bot.reply_to(msg, "⚠️ Сессия уже идёт. Используй /stop_listen чтобы завершить.")
            return
        listening_active[chat_id] = True
        listening_sessions[chat_id] = []

    bot.reply_to(
        msg,
        "🟢 *Запись сессии начата!*\n"
        "Все сообщения будут сохранены и проанализированы.\n"
        "Отправь /stop_listen чтобы завершить и получить резюме.",
    )
    logger.info("Listening started in chat %d", chat_id)


# ══════════════════════════════════════════════════════════════════════════════
#  /stop_listen  — end session, summarise, index
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(commands=["stop_listen"])
def cmd_stop_listen(msg: Message):
    chat_id = msg.chat.id
    with _lock:
        if not is_listening(chat_id):
            bot.reply_to(msg, "⚠️ Активной сессии нет. Начни с /start_listen.")
            return
        listening_active[chat_id] = False
        session = listening_sessions.pop(chat_id, [])

    if not session:
        bot.reply_to(msg, "ℹ️ Сессия завершена, но сообщений не было.")
        return

    # Build a readable dialogue string
    dialogue_lines = []
    haystack_docs = []
    for entry in session:
        line = f"[{entry['timestamp']}] {entry['author_name']}: {entry['text']}"
        dialogue_lines.append(line)
        haystack_docs.append(
            Document(
                content=entry["text"],
                meta={
                    "chat_id": str(chat_id),
                    "author_id": str(entry["author_id"]),
                    "author_name": entry["author_name"],
                    "timestamp": entry["timestamp"],
                    "type": "session_message",
                },
            )
        )

    dialogue_text = "\n".join(dialogue_lines)

    bot.send_message(chat_id, "⏳ Анализирую диалог, подождите...")

    # Index session messages into Pinecone in background
    threading.Thread(target=index_documents_batch, args=(haystack_docs,), daemon=True).start()

    # Summarise via OpenAI
    try:
        result = summary_pipeline.run(
            {"prompt_builder": {"dialogue": dialogue_text}}
        )
        summary = result["generator"]["replies"][0]
    except Exception as e:
        summary = f"❌ Ошибка генерации резюме: {e}"

    bot.send_message(
        chat_id,
        f"🏁 *Сессия завершена!* ({len(session)} сообщений)\n\n{summary}",
    )
    logger.info("Session stopped in chat %d, %d messages processed", chat_id, len(session))


# ══════════════════════════════════════════════════════════════════════════════
#  /summarise  — summarise entire chat history from Pinecone
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(commands=["summarise", "summarize"])
def cmd_summarise(msg: Message):
    chat_id = msg.chat.id
    bot.send_message(chat_id, "⏳ Собираю историю чата из базы данных...")

    try:
        docs = document_store.filter_documents(
            filters={"field": "meta.chat_id", "operator": "==", "value": str(chat_id)}
        )
    except Exception as e:
        bot.reply_to(msg, f"❌ Ошибка при чтении базы: {e}")
        return

    if not docs:
        bot.reply_to(msg, "ℹ️ История чата пуста — нечего резюмировать.")
        return

    docs_sorted = sorted(docs, key=lambda d: d.meta.get("timestamp", ""))
    dialogue_lines = [
        f"[{d.meta.get('timestamp','')}] {d.meta.get('author_name','?')}: {d.content}"
        for d in docs_sorted[:200]  # cap to avoid token overflow
    ]
    dialogue_text = "\n".join(dialogue_lines)

    try:
        result = summary_pipeline.run({"prompt_builder": {"dialogue": dialogue_text}})
        summary = result["generator"]["replies"][0]
    except Exception as e:
        summary = f"❌ Ошибка генерации: {e}"

    bot.send_message(chat_id, f"📊 *Резюме чата:*\n\n{summary}")


# ══════════════════════════════════════════════════════════════════════════════
#  /help
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(commands=["start", "help"])
def cmd_help(msg: Message):
    bot.reply_to(
        msg,
        "👋 *Team Assistant Bot*\n\n"
        "Я слушаю чат, сохраняю историю в векторную БД и помогаю команде.\n\n"
        "*Команды:*\n"
        "/start_listen — начать запись сессии обсуждения\n"
        "/stop_listen — завершить сессию и получить резюме + вердикт\n"
        "/summarise — резюме всего чата из базы данных\n"
        "/help — эта справка\n\n"
        "*Интерактив:*\n"
        "• Упомяни меня (`@bot`) и задай вопрос — отвечу по контексту чата\n"
        "• Отправь голосовое сообщение — транскрибирую и сохраню в память\n"
        "• Загрузи аудиофайл (MP3, WAV, M4A…) — распознаю речь\n"
        "• Загрузи документ (PDF, DOCX, TXT) — проанализирую\n\n"
        f"Whisper: ✅ whisper-1 через proxyapi.ru\n"
        f"pydub (OGG→MP3): {'✅' if PYDUB_AVAILABLE else '⚠️ pip install pydub + ffmpeg'}\n"
        f"Docling: {'✅' if DOCLING_AVAILABLE else '⚠️ pip install docling'}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Voice message handler (Telegram voice notes, OGG/OPUS)
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(content_types=["voice"])
def handle_voice(msg: Message):
    """Handle Telegram voice messages (🎤 hold-to-record button)."""
    voice = msg.voice
    if voice is None:
        return
    threading.Thread(
        target=_download_and_transcribe,
        args=(msg, voice.file_id, voice.duration or 0, True, msg.caption or ""),
        daemon=True,
    ).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Audio file handler (MP3, WAV, M4A, OGG sent as files or music)
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(content_types=["audio"])
def handle_audio(msg: Message):
    """Handle audio files sent to the chat (music attach or audio file)."""
    audio = msg.audio
    if audio is None:
        return
    mime = audio.mime_type or ""
    # Accept only speech-like audio; skip music tracks without speech intent
    # (user can always send as document if MIME is unexpected)
    if mime and mime not in AUDIO_MIME_SUPPORTED and not mime.startswith("audio/"):
        bot.reply_to(
            msg,
            "⚠️ Формат аудиофайла не поддерживается для транскрипции.\n"
            "Поддерживаются: MP3, MP4, WAV, M4A, OGG, WebM.",
        )
        return
    threading.Thread(
        target=_download_and_transcribe,
        args=(msg, audio.file_id, audio.duration or 0, False, msg.caption or ""),
        daemon=True,
    ).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Document handler — also catches audio files sent as documents
# ══════════════════════════════════════════════════════════════════════════════

def _is_bot_mention(msg: Message) -> bool:
    """Return True if the bot is @mentioned or replied-to in this message.

    Checks three ways:
    1. Telegram entities of type 'mention' that match the bot username.
    2. Plain text substring @username (fallback for clients that omit entities).
    3. The message is a reply to a bot message.
    """
    username = get_bot_username()
    if not username:
        # Can't determine username yet — skip to avoid false positives
        return False

    mention_str = f"@{username}"

    # 1. Check Telegram mention entities (most reliable)
    entities = msg.entities or []
    text = msg.text or ""
    for ent in entities:
        if ent.type == "mention":
            # entity offset/length are in UTF-16 code units; use text slicing
            mentioned = text[ent.offset: ent.offset + ent.length]
            if mentioned.lower() == mention_str.lower():
                return True

    # 2. Plain-text fallback (case-insensitive)
    if mention_str.lower() in text.lower():
        return True

    # 3. Reply to a bot message
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if (msg.reply_to_message.from_user.username or "").lower() == username.lower():
            return True

    return False


def _extract_question(msg: Message) -> str:
    """Extract the question text, stripping the bot mention."""
    username = get_bot_username()
    text = msg.text or msg.caption or ""
    if username:
        # Remove @username (case-insensitive)
        text = re.sub(re.escape(f"@{username}"), "", text, flags=re.IGNORECASE).strip()
    return text or "Что происходит в этом чате?"


@bot.message_handler(
    content_types=["text"],
    func=lambda m: m.chat.type in ("group", "supergroup", "private"),
)
def handle_text(msg: Message):
    chat_id = msg.chat.id
    author_id = msg.from_user.id if msg.from_user else 0
    author_name = format_author(msg)
    text = msg.text or ""
    timestamp = datetime.fromtimestamp(msg.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Always index into Pinecone (synchronous for RAG accuracy)
    index_message(chat_id, author_id, author_name, text, timestamp)

    # 2. If active session — record in memory buffer too
    if is_listening(chat_id):
        with _lock:
            if chat_id in listening_sessions:
                listening_sessions[chat_id].append(
                    {
                        "author_id": author_id,
                        "author_name": author_name,
                        "text": text,
                        "timestamp": timestamp,
                    }
                )

    # 3. If bot is mentioned — answer via RAG
    if _is_bot_mention(msg):
        question = _extract_question(msg)
        logger.info("Bot mentioned in chat %d, question: %s", chat_id, question[:80])
        bot.send_chat_action(chat_id, "typing")
        try:
            result = rag_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "retriever": {"filters": build_chat_filter(chat_id)},
                    "prompt_builder": {"query": question},
                }
            )
            answer = result["generator"]["replies"][0]
        except Exception as e:
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            answer = f"❌ Не удалось получить ответ: {e}"
        bot.reply_to(msg, answer)


# ══════════════════════════════════════════════════════════════════════════════
#  Document handler (PDF / DOCX / TXT)
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
}


def _process_and_reply(msg: Message, file_id: str, file_name: str, caption: str):
    chat_id = msg.chat.id
    author_name = format_author(msg)

    bot.send_chat_action(chat_id, "upload_document")
    bot.reply_to(msg, f"📥 Получил файл *{file_name}*. Обрабатываю через Docling…")

    try:
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
    except Exception as e:
        bot.reply_to(msg, f"❌ Не удалось скачать файл: {e}")
        return

    suffix = Path(file_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(downloaded)
        tmp_path = tmp.name

    try:
        doc_text = process_document_with_docling(tmp_path)
        if not doc_text:
            bot.reply_to(msg, "❌ Не удалось извлечь текст из документа.")
            return

        # Determine task from caption
        task = caption.strip() if caption else "Дай полное резюме документа."

        bot.send_chat_action(chat_id, "typing")
        result = doc_analysis_pipeline.run(
            {"prompt_builder": {"doc_content": doc_text[:12000], "task": task}}
        )
        analysis = result["generator"]["replies"][0]

        # Index document into Pinecone
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        doc = Document(
            content=doc_text,
            meta={
                "chat_id": str(chat_id),
                "author_name": author_name,
                "file_name": file_name,
                "timestamp": timestamp,
                "type": "uploaded_document",
            },
        )
        threading.Thread(target=index_documents_batch, args=([doc],), daemon=True).start()

        bot.send_message(
            chat_id,
            f"📄 *Анализ документа «{file_name}»:*\n\n{analysis}",
        )

    except Exception as e:
        bot.reply_to(msg, f"❌ Ошибка анализа: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@bot.message_handler(content_types=["document"])
def handle_document(msg: Message):
    doc = msg.document
    if doc is None:
        return

    file_name = doc.file_name or "document"
    mime = doc.mime_type or ""
    caption = msg.caption or ""

    # ── Route audio files sent as documents to Whisper ──────────────────────
    audio_extensions = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".webm", ".mp4", ".mpga")
    if mime.startswith("audio/") or any(file_name.lower().endswith(ext) for ext in audio_extensions):
        threading.Thread(
            target=_download_and_transcribe,
            args=(msg, doc.file_id, 0, False, caption),
            daemon=True,
        ).start()
        return

    # ── Route documents to Docling analysis ─────────────────────────────────
    if mime not in SUPPORTED_MIME and not any(
        file_name.lower().endswith(ext) for ext in (".pdf", ".docx", ".doc", ".txt", ".md")
    ):
        bot.reply_to(
            msg,
            "⚠️ Поддерживаются:\n"
            "• Документы: PDF, DOCX, DOC, TXT, MD\n"
            "• Аудио: MP3, WAV, M4A, OGG, WebM\n"
            "Добавь подпись с вопросом или задачей.",
        )
        return

    threading.Thread(
        target=_process_and_reply,
        args=(msg, doc.file_id, file_name, caption),
        daemon=True,
    ).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Bot starting…")
    get_bot_username()
    logger.info("Bot username: @%s", BOT_USERNAME)
    logger.info("Pinecone index: %s | Embedding model: %s", PINECONE_INDEX_NAME, EMBEDDING_MODEL)
    logger.info("OpenAI base URL: %s | Chat model: %s", OPENAI_BASE_URL, OPENAI_MODEL)
    logger.info("Whisper model: %s | pydub: %s | Docling: %s",
                WHISPER_MODEL,
                "✓" if PYDUB_AVAILABLE else "✗ (install pydub + ffmpeg)",
                "✓" if DOCLING_AVAILABLE else "✗ (install docling)")
    logger.info(
        "IMPORTANT: For the bot to read ALL group messages (not just commands/mentions), "
        "disable Privacy Mode in @BotFather: /mybots → your bot → Bot Settings → Group Privacy → Turn off"
    )
    # allowed_updates ensures we receive message updates including text with mentions
    bot.infinity_polling(logger_level=logging.WARNING, allowed_updates=["message", "edited_message"])
