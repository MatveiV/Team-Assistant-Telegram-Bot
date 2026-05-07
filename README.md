# 🤖 Team Assistant Telegram Bot

Умный командный Telegram-бот на базе **Haystack 2.x**, **Pinecone** и **OpenAI** (через [proxyapi.ru](https://proxyapi.ru)).  
Понимает текст, голосовые сообщения и документы. Сохраняет всю историю команды в векторную БД, умеет резюмировать обсуждения и отвечать на вопросы по контексту переписки.

---

## Возможности

| Функция | Описание |
|---|---|
| 📡 Пассивное слушание | Каждое текстовое сообщение автоматически индексируется в Pinecone |
| 🎙 Голосовые сообщения | Транскрибация через Whisper → индексация текста |
| 🎵 Аудиофайлы | MP3, WAV, M4A, OGG, WebM → Whisper → текст → Pinecone |
| 🟢 `/start_listen` | Начать активную запись сессии обсуждения |
| 🔴 `/stop_listen` | Завершить сессию → резюме + вердикт от ИИ |
| 💬 Упоминание `@team_assistant_text_audio_bot` | RAG-ответ на вопрос по всей истории чата |
| 📄 Документы | Анализ PDF / DOCX / TXT через Docling |
| 📊 `/summarise` | Глобальное резюме чата из векторной БД |

---

## Быстрый старт

### 1. Клонируй и настрой окружение

```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Для аудио** дополнительно нужен системный `ffmpeg`:
> ```bash
> apt install ffmpeg       # Ubuntu / Debian
> brew install ffmpeg      # macOS
> # Windows — скачать с https://ffmpeg.org/download.html и добавить в PATH
> ```

### 2. Создай `.env`

```bash
cp .env.example .env
# Заполни значения в .env
```

### 3. Переменные окружения

| Переменная | Обязательна | Описание |
|---|:---:|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | Токен бота от [@BotFather](https://t.me/BotFather) |
| `PINECONE_API_KEY` | ✅ | API-ключ [Pinecone](https://console.pinecone.io) |
| `PINECONE_INDEX_NAME` | — | Имя индекса (default: `team-assistant`) |
| `OPENAI_BASE_URL` | — | `https://api.proxyapi.ru/openai/v1` |
| `OPENAI_API_KEY` | ✅ | API-ключ для proxyapi.ru |
| `OPENAI_MODEL` | — | Модель генерации (default: `gpt-4o`) |
| `WHISPER_MODEL` | — | Модель транскрипции (default: `whisper-1`) |
| `EMBEDDING_MODEL` | — | Модель эмбеддингов (default: `text-embedding-3-small`) |

### 4. Настрой бота в Telegram

> ⚠️ **Критически важно:** без отключения Privacy Mode бот **не будет получать обычные сообщения** в группе и не сможет отвечать на упоминания `@team_assistant_text_audio_bot`.

1. Открой [@BotFather](https://t.me/BotFather) и выполни одно из:
   - **Новый способ:** `/mybots` → выбери бота → **Bot Settings** → **Group Privacy** → **Turn off**
   - **Старый способ:** `/setprivacy` → выбери бота → **Disable**
2. Добавь бота в рабочую группу с правами на чтение сообщений
3. Для вызова бота или вопроса упомяни его: `@team_assistant_text_audio_bot [ваш вопрос]`
4. После добавления в группу бот автоматически начнёт индексировать все сообщения

### 5. Запусти

```bash
python bot.py
```

---

## Диаграммы архитектуры

### C4 Level 1 — Системный контекст

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#64748b', 'secondaryColor': '#f0fdf4', 'tertiaryColor': '#fefce8'}}}%%
C4Context
    title System Context — Team Assistant Bot

    Person(user, "Участник команды", "Пишет в Telegram-группу,\nотправляет голос и документы")

    System(bot, "Team Assistant Bot", "Telegram-бот на Python.\nПонимает текст, аудио, документы.\nОтвечает на вопросы по истории чата.")

    System_Ext(telegram, "Telegram API", "Доставка сообщений,\nголосовых заметок,\nфайлов")
    System_Ext(openai, "OpenAI API\n(via proxyapi.ru)", "GPT-4o — генерация ответов\nWhisper-1 — транскрипция аудио\ntext-embedding — векторизация")
    System_Ext(pinecone, "Pinecone", "Облачная векторная БД.\nХранит эмбеддинги сообщений,\nаудио-транскриптов, документов")

    Rel(user, telegram, "Отправляет сообщения,\nголос, файлы")
    Rel(telegram, bot, "Webhook / long polling")
    Rel(bot, openai, "Эмбеддинги, генерация,\nтранскрипция (HTTPS)")
    Rel(bot, pinecone, "Запись и поиск\nвекторов (HTTPS)")
    Rel(bot, telegram, "Отправляет ответы\nи транскрипты")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

### C4 Level 2 — Контейнеры

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#64748b', 'secondaryColor': '#f0fdf4', 'tertiaryColor': '#fefce8', 'noteBkgColor': '#fefce8', 'noteTextColor': '#713f12'}}}%%
C4Container
    title Container Diagram — Team Assistant Bot

    Person(user, "Участник команды", "Telegram-клиент")

    System_Boundary(botSystem, "Team Assistant Bot (Python процесс)") {
        Container(dispatcher, "Telegram Dispatcher", "pyTelegramBotAPI", "Маршрутизирует входящие\nсообщения по типу:\ntext / voice / audio / document")

        Container(pipelines, "Haystack Pipelines", "haystack-ai 2.x", "5 конвейеров:\n• Indexing\n• RAG\n• Transcription\n• Summary\n• Doc Analysis")

        Container(docling, "Docling Converter", "docling", "Конвертирует PDF/DOCX/TXT\nв текст для анализа")

        Container(pydub, "Audio Converter", "pydub + ffmpeg", "OGG/OPUS -> MP3\nдля Whisper API")

        ContainerDb(state, "In-Memory State", "Python dict", "Буфер активных сессий:\nlistening_sessions,\nlistening_active")
    }

    System_Ext(telegram, "Telegram API", "Long polling")
    System_Ext(openaiEmbed, "OpenAI Embeddings\n(proxyapi.ru)", "text-embedding-3-small/large")
    System_Ext(openaiGPT, "OpenAI Chat\n(proxyapi.ru)", "gpt-4o")
    System_Ext(openaiWhisper, "OpenAI Whisper\n(proxyapi.ru)", "whisper-1")
    System_Ext(pinecone, "Pinecone", "Serverless vector DB")

    Rel(user, telegram, "Сообщения / файлы")
    Rel(telegram, dispatcher, "Updates")
    Rel(dispatcher, pipelines, "Текст, путь к файлу,\nвопрос пользователя")
    Rel(dispatcher, pydub, "OGG-файл")
    Rel(dispatcher, docling, "PDF / DOCX / TXT")
    Rel(pydub, pipelines, "MP3-файл")
    Rel(docling, pipelines, "Извлечённый текст")
    Rel(pipelines, state, "Чтение/запись\nбуфера сессии")
    Rel(pipelines, openaiEmbed, "Запрос эмбеддингов")
    Rel(pipelines, openaiGPT, "Генерация ответа")
    Rel(pipelines, openaiWhisper, "Транскрипция аудио")
    Rel(pipelines, pinecone, "Запись / поиск векторов")
    Rel(pipelines, telegram, "Готовый ответ")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

### C4 Level 3 — Компоненты Haystack Pipelines

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#64748b', 'secondaryColor': '#f0fdf4', 'tertiaryColor': '#fefce8'}}}%%
C4Component
    title Component Diagram — Haystack Pipelines

    System_Boundary(indexing, "Indexing Pipeline") {
        Component(docEmbed, "OpenAIDocumentEmbedder", "haystack", "Векторизует текст\n(1536 / 3072 dim)")
        Component(writer, "DocumentWriter", "haystack", "Записывает документы\nв PineconeDocumentStore")
    }

    System_Boundary(transcription, "Transcription Pipeline") {
        Component(whisper, "RemoteWhisperTranscriber", "haystack", "Отправляет MP3 в Whisper API,\nвозвращает Document с текстом")
    }

    System_Boundary(rag, "RAG Pipeline") {
        Component(textEmbed, "OpenAITextEmbedder", "haystack", "Векторизует запрос")
        Component(retriever, "PineconeEmbeddingRetriever", "haystack-pinecone", "ANN-поиск top-8 документов")
        Component(promptRAG, "PromptBuilder", "haystack", "Jinja2-шаблон:\nконтекст + вопрос")
        Component(generatorRAG, "OpenAIGenerator", "haystack", "GPT-4o - ответ со\nссылками на авторов")
    }

    System_Boundary(summary, "Summary Pipeline") {
        Component(promptSum, "PromptBuilder", "haystack", "Шаблон: анализ диалога,\nспор / решение / итог")
        Component(generatorSum, "OpenAIGenerator", "haystack", "GPT-4o - резюме,\nвердикт, action-items")
    }

    System_Boundary(docAnalysis, "Doc Analysis Pipeline") {
        Component(promptDoc, "PromptBuilder", "haystack", "Шаблон: контент\nдокумента + задача")
        Component(generatorDoc, "OpenAIGenerator", "haystack", "GPT-4o - анализ\nдокумента")
    }

    ContainerDb(pineconeDB, "Pinecone", "Vector DB", "Индекс: team-assistant\nNamespace: chat-history")

    Rel(docEmbed, writer, "documents (с эмбеддингами)")
    Rel(writer, pineconeDB, "write_documents()")
    Rel(whisper, docEmbed, "Document(content=transcript)")
    Rel(textEmbed, retriever, "query_embedding")
    Rel(retriever, pineconeDB, "ANN query")
    Rel(retriever, promptRAG, "documents")
    Rel(promptRAG, generatorRAG, "prompt")
    Rel(promptSum, generatorSum, "prompt")
    Rel(promptDoc, generatorDoc, "prompt")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

### UML — Диаграмма последовательности: голосовое сообщение

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#2563eb', 'lineColor': '#475569', 'signalColor': '#475569', 'signalTextColor': '#1e293b', 'labelBoxBkgColor': '#dbeafe', 'labelBoxBorderColor': '#3b82f6', 'labelTextColor': '#1e3a5f', 'loopTextColor': '#1e3a5f', 'noteBkgColor': '#fefce8', 'noteTextColor': '#713f12', 'noteBorderColor': '#ca8a04', 'activationBkgColor': '#eff6ff', 'activationBorderColor': '#3b82f6', 'sequenceNumberColor': '#ffffff'}}}%%
sequenceDiagram
    autonumber
    actor User as Участник
    participant TG as Telegram API
    participant Disp as Dispatcher
    participant Conv as Audio Converter
    participant Pipe as Transcription Pipeline
    participant Whisper as Whisper API
    participant Idx as Indexing Pipeline
    participant Embed as Embeddings API
    participant PC as Pinecone

    User->>TG: Голосовое сообщение (OGG/OPUS)
    TG->>Disp: Update{voice, file_id, duration}
    Disp->>Disp: handle_voice() запускает поток
    Disp-->>TG: Транскрибирую...
    TG-->>User: Уведомление

    Note over Disp,Conv: Фоновый поток
    Disp->>TG: getFile(file_id)
    TG-->>Disp: file_path
    Disp->>TG: downloadFile(file_path)
    TG-->>Disp: bytes (OGG)
    Disp->>Conv: ogg_bytes
    Conv->>Conv: AudioSegment.from_file(ogg).export(mp3)
    Conv-->>Disp: tmp.mp3

    Disp->>Pipe: run({sources:[tmp.mp3]})
    Pipe->>Whisper: POST /audio/transcriptions (model=whisper-1)
    Whisper-->>Pipe: {text: транскрипт}
    Pipe-->>Disp: Document(content=transcript)

    Disp->>TG: sendMessage(Транскрипция: ...)
    TG-->>User: Транскрипт в чате

    par Индексация в Pinecone
        Disp->>Idx: run({documents:[Document+meta]})
        Idx->>Embed: POST /embeddings
        Embed-->>Idx: embedding[1536]
        Idx->>PC: upsert(vector, metadata)
    and Запись в буфер сессии
        Disp->>Disp: sessions[chat_id].append(...)
    end

    Note over Conv,PC: Временные файлы OGG/MP3 удаляются
```

---

### UML — Диаграмма последовательности: RAG-запрос (@упоминание)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#2563eb', 'lineColor': '#475569', 'signalColor': '#475569', 'signalTextColor': '#1e293b', 'labelBoxBkgColor': '#dbeafe', 'labelBoxBorderColor': '#3b82f6', 'labelTextColor': '#1e3a5f', 'loopTextColor': '#1e3a5f', 'noteBkgColor': '#fefce8', 'noteTextColor': '#713f12', 'noteBorderColor': '#ca8a04', 'activationBkgColor': '#eff6ff', 'activationBorderColor': '#3b82f6', 'sequenceNumberColor': '#ffffff'}}}%%
sequenceDiagram
    autonumber
    actor User as Участник
    participant TG as Telegram API
    participant Disp as Dispatcher
    participant RAG as RAG Pipeline
    participant Embed as Embeddings API
    participant PC as Pinecone
    participant GPT as GPT-4o

    User->>TG: @bot Что решили про дедлайн релиза?
    TG->>Disp: Update{text, mention=true}
    Disp->>Disp: _is_bot_mention() = True
    Disp->>TG: sendChatAction(typing)

    Disp->>RAG: run({text_embedder:{text:question}, prompt_builder:{query:question}})
    RAG->>Embed: POST /embeddings (text=question)
    Embed-->>RAG: query_embedding[1536]

    RAG->>PC: query(embedding, top_k=8)
    PC-->>RAG: [{content, meta{author,timestamp}}, ...]

    RAG->>RAG: PromptBuilder заполняет шаблон

    RAG->>GPT: POST /chat/completions (prompt)
    GPT-->>RAG: Ответ со ссылками на авторов

    RAG-->>Disp: answer text
    Disp->>TG: replyTo(msg, answer)
    TG-->>User: Ответ в чате
```

---

### UML — Диаграмма последовательности: сессия /start_listen → /stop_listen

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#2563eb', 'lineColor': '#475569', 'signalColor': '#475569', 'signalTextColor': '#1e293b', 'labelBoxBkgColor': '#dbeafe', 'labelBoxBorderColor': '#3b82f6', 'labelTextColor': '#1e3a5f', 'loopTextColor': '#1e3a5f', 'noteBkgColor': '#fefce8', 'noteTextColor': '#713f12', 'noteBorderColor': '#ca8a04', 'activationBkgColor': '#eff6ff', 'activationBorderColor': '#3b82f6', 'sequenceNumberColor': '#ffffff'}}}%%
sequenceDiagram
    autonumber
    actor Team as Команда
    participant TG as Telegram API
    participant Disp as Dispatcher
    participant Mem as In-Memory State
    participant IdxP as Indexing Pipeline
    participant SumP as Summary Pipeline
    participant PC as Pinecone
    participant GPT as GPT-4o

    Team->>TG: /start_listen
    TG->>Disp: Update{command=start_listen}
    Disp->>Mem: listening_active[chat_id]=True, sessions[chat_id]=[]
    Disp->>TG: Запись сессии начата
    TG-->>Team: Уведомление

    loop Команда общается
        Team->>TG: Сообщение / голос / документ
        TG->>Disp: Update
        Disp->>Mem: sessions[chat_id].append({author, text, ts})
        Disp->>IdxP: (фоново) индексировать
        IdxP->>PC: upsert
    end

    Team->>TG: /stop_listen
    TG->>Disp: Update{command=stop_listen}
    Disp->>Mem: listening_active=False, session=sessions.pop()
    Disp->>TG: Анализирую диалог...

    par Индексация всей сессии
        Disp->>IdxP: index_documents_batch(session_docs)
        IdxP->>PC: upsert all
    and Генерация резюме
        Disp->>SumP: run({dialogue: текст_диалога})
        SumP->>GPT: prompt (анализ спора / решения)
        GPT-->>SumP: резюме + вердикт + action-items
        SumP-->>Disp: summary text
    end

    Disp->>TG: Сессия завершена + резюме
    TG-->>Team: Итоговый анализ диалога
```

---

### UML — Диаграмма состояний бота

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#475569', 'edgeLabelBackground': '#f8fafc', 'tertiaryColor': '#f0fdf4', 'noteBkgColor': '#fefce8', 'noteTextColor': '#713f12'}}}%%
stateDiagram-v2
    direction LR

    [*] --> Idle : Запуск бота

    state Idle {
        [*] --> PassiveListening
        PassiveListening : Пассивный режим\nИндексирует все сообщения\nи аудио в Pinecone
    }

    state ActiveSession {
        [*] --> Recording
        Recording : Запись сессии\nБуфер + индексация
        Recording --> Recording : текст / голос / аудио / документ
    }

    state Summarising {
        [*] --> BatchIndexing
        BatchIndexing : index_documents_batch()\nвсе сообщения сессии в Pinecone
        BatchIndexing --> Generating
        Generating : Summary Pipeline\nGPT-4o анализирует диалог
        Generating --> [*]
    }

    state Answering {
        [*] --> Retrieving
        Retrieving : RAG Pipeline\nPinecone top-8 chunks
        Retrieving --> GeneratingAnswer
        GeneratingAnswer : GPT-4o формирует ответ\nсо ссылками на авторов
        GeneratingAnswer --> [*]
    }

    Idle --> ActiveSession : /start_listen
    ActiveSession --> Summarising : /stop_listen
    Summarising --> Idle : Резюме отправлено

    Idle --> Answering : @упоминание бота
    ActiveSession --> Answering : @упоминание бота
    Answering --> Idle : Ответ отправлен (нет сессии)
    Answering --> ActiveSession : Ответ отправлен (сессия активна)

    note right of Idle
        В любом состоянии:
        Голос/Аудио - Whisper - текст
        Документ - Docling - анализ
        /summarise - резюме из БД
    end note
```

---

### UML — Диаграмма классов

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#475569', 'classText': '#1e3a5f'}}}%%
classDiagram
    direction TB

    class TelegramDispatcher {
        +BOT_USERNAME : str
        +listening_active : dict~int,bool~
        +listening_sessions : dict~int,list~
        +handle_text(msg : Message)
        +handle_voice(msg : Message)
        +handle_audio(msg : Message)
        +handle_document(msg : Message)
        +cmd_start_listen(msg : Message)
        +cmd_stop_listen(msg : Message)
        +cmd_summarise(msg : Message)
        +cmd_help(msg : Message)
        -_is_bot_mention(msg) bool
        -_extract_question(msg) str
        -format_author(msg) str
    }

    class IndexingPipeline {
        <<Haystack Pipeline>>
        +OpenAIDocumentEmbedder
        +DocumentWriter
        +run(documents) dict
    }

    class TranscriptionPipeline {
        <<Haystack Pipeline>>
        +RemoteWhisperTranscriber
        +run(sources) dict
    }

    class RAGPipeline {
        <<Haystack Pipeline>>
        +OpenAITextEmbedder
        +PineconeEmbeddingRetriever
        +PromptBuilder
        +OpenAIGenerator
        +run(query) dict
    }

    class SummaryPipeline {
        <<Haystack Pipeline>>
        +PromptBuilder
        +OpenAIGenerator
        +run(dialogue) dict
    }

    class DocAnalysisPipeline {
        <<Haystack Pipeline>>
        +PromptBuilder
        +OpenAIGenerator
        +run(doc_content, task) dict
    }

    class AudioConverter {
        <<pydub + ffmpeg>>
        +convert_ogg_to_mp3(path : str) str
    }

    class DoclingConverter {
        <<docling>>
        +process_document(path : str) str
    }

    class PineconeDocumentStore {
        <<haystack-pinecone>>
        +index : str
        +namespace : str
        +dimension : int
        +metric : str
        +write_documents(docs : list)
        +filter_documents(filters : dict) list
    }

    class OpenAIBackend {
        <<proxyapi.ru>>
        +base_url : str
        +chat_completions(prompt) str
        +embeddings(text) list~float~
        +audio_transcriptions(file) str
    }

    TelegramDispatcher --> IndexingPipeline : index_message()
    TelegramDispatcher --> TranscriptionPipeline : transcribe_audio_file()
    TelegramDispatcher --> RAGPipeline : rag_pipeline.run()
    TelegramDispatcher --> SummaryPipeline : summary_pipeline.run()
    TelegramDispatcher --> DocAnalysisPipeline : doc_analysis_pipeline.run()
    TelegramDispatcher --> AudioConverter : _convert_ogg_to_mp3()
    TelegramDispatcher --> DoclingConverter : process_document_with_docling()

    IndexingPipeline --> PineconeDocumentStore : write
    RAGPipeline --> PineconeDocumentStore : query
    SummaryPipeline --> OpenAIBackend : chat
    DocAnalysisPipeline --> OpenAIBackend : chat
    IndexingPipeline --> OpenAIBackend : embeddings
    RAGPipeline --> OpenAIBackend : embeddings + chat
    TranscriptionPipeline --> OpenAIBackend : whisper
```

---

### UML — Диаграмма развёртывания

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#dbeafe', 'primaryTextColor': '#1e3a5f', 'primaryBorderColor': '#3b82f6', 'lineColor': '#475569', 'clusterBkg': '#f8fafc', 'clusterBorder': '#94a3b8', 'edgeLabelBackground': '#ffffff'}}}%%
graph TD
    subgraph users["Пользователи"]
        U1["Участник 1\niPhone / Android"]
        U2["Участник 2\nDesktop"]
        U3["Участник N"]
    end

    subgraph telegram["Telegram Infrastructure"]
        TG["Telegram Servers\napi.telegram.org"]
    end

    subgraph server["Сервер / VPS"]
        direction TB
        BOT["bot.py\nPython 3.11+"]
        FFMPEG["ffmpeg\nсистемный бинарь"]
        ENV["env\nПеременные окружения"]
        TMP["tmp\nВременные аудиофайлы"]
        BOT --> FFMPEG
        BOT -.-> ENV
        BOT --> TMP
    end

    subgraph openai_proxy["proxyapi.ru → OpenAI"]
        PROXY["api.proxyapi.ru/openai/v1"]
        GPT4["GPT-4o\nChat Completions"]
        WHISPER["Whisper-1\nAudio Transcriptions"]
        EMBED["text-embedding-3-small\nEmbeddings"]
        PROXY --> GPT4
        PROXY --> WHISPER
        PROXY --> EMBED
    end

    subgraph pinecone_cloud["Pinecone Serverless\nAWS us-east-1"]
        PCIDX["Index: team-assistant\nNamespace: chat-history\nDimension: 1536\nMetric: cosine"]
    end

    U1 & U2 & U3 -->|HTTPS| TG
    TG -->|"Long polling getUpdates"| BOT
    BOT -->|"sendMessage / replyTo"| TG

    BOT -->|"embeddings / chat / whisper HTTPS"| PROXY
    BOT -->|"upsert / query HTTPS"| PCIDX

    style server fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
    style telegram fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style openai_proxy fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style pinecone_cloud fill:#fce7f3,stroke:#ec4899,stroke-width:2px
    style users fill:#f8fafc,stroke:#94a3b8,stroke-width:2px
```

---

## Использование в чате

### Запись обсуждения

```
/start_listen
  Алексей: давайте перенесём дедлайн на следующую пятницу
  Мария: нет, заказчик не согласится
  🎤 [Иван голосовое]: я думаю компромисс — сделать MVP к среде
/stop_listen
```
→ Бот выдаст: резюме спора, вердикт, action-items с ответственными.

### Вопрос по истории чата

```
@your_bot_name Что решили по поводу дедлайна релиза?
```
→ Бот находит релевантные фрагменты в Pinecone и отвечает со ссылками на авторов и время.

### Голосовые сообщения и аудиофайлы

Отправь голосовое 🎤 или прикрепи MP3/WAV/M4A — бот автоматически:
1. Транскрибирует через Whisper
2. Покажет транскрипт в чате
3. Сохранит в векторную БД как обычное сообщение

### Анализ документа

Загрузи PDF / DOCX / TXT с подписью:
```
Выдели все ключевые требования из этого ТЗ
```
Без подписи — бот сделает полное резюме документа.

### Глобальное резюме

```
/summarise
```
→ Бот соберёт до 200 последних сообщений из Pinecone и сформирует сводку.

---

## Технический стек

| Слой | Технология |
|---|---|
| Бот | pyTelegramBotAPI 4.x (sync, long polling) |
| AI-оркестрация | Haystack 2.x Pipeline API |
| Векторная БД | Pinecone Serverless (AWS us-east-1) |
| Генерация | OpenAI GPT-4o via proxyapi.ru |
| Транскрипция | OpenAI Whisper-1 via proxyapi.ru |
| Эмбеддинги | text-embedding-3-small (1536d) |
| Документы | Docling (PDF/DOCX/TXT → Markdown) |
| Аудио-конвертация | pydub + ffmpeg (OGG → MP3) |

### Поддерживаемые форматы

| Тип | Форматы |
|---|---|
| Документы | PDF, DOCX, DOC, TXT, MD |
| Аудиофайлы | MP3, WAV, M4A, OGG, WebM, MP4 |
| Голос Telegram | OGG/OPUS (автоконвертация в MP3) |

### Ограничения

- Максимальная длительность аудио: **10 минут** (600 сек) — настраивается в `MAX_DURATION_SECONDS`
- Резюме чата: до **200 последних документов** из Pinecone
- Анализ документов: до **12 000 символов** на вызов GPT

---

## Устранение неполадок

### Бот не отвечает на `@упоминание` в группе

Самая частая причина — включён **Privacy Mode** (включён по умолчанию для всех новых ботов).

| Симптом | Причина | Решение |
|---|---|---|
| Бот не реагирует на `@bot вопрос` | Privacy Mode включён | `/mybots` → Bot Settings → Group Privacy → **Turn off** |
| Бот отвечает в личке, но не в группе | Та же причина | То же решение |
| Бот отвечает на команды (`/help`), но не на `@упоминание` | Privacy Mode включён | То же решение |

После изменения настройки **перезапусти бота** и **удали/добавь его в группу заново** (Telegram кэширует права).

### Бот не видит историю чата при `@упоминании`

Если Pinecone пустой (бот только что добавлен), ответ будет основан на пустом контексте. Нужно:
1. Написать несколько сообщений в группе — они проиндексируются автоматически
2. Или запустить сессию `/start_listen` → обсудить тему → `/stop_listen`

### Ошибка при транскрипции аудио

```
OGG→MP3 conversion failed
```
Убедись, что `ffmpeg` установлен и путь указан в `FFMPEG_PATH` (Windows) или доступен в `PATH` (Linux/macOS).

---

## Примечания

- **Pinecone**: индекс создаётся автоматически при первом запуске, бесплатный serverless-тир совместим
- **proxyapi.ru**: подставляется через `api_base_url` в каждом Haystack-компоненте — GPT, Whisper, Embeddings
- **Параллельность**: индексация в Pinecone выполняется в фоновых потоках, не блокируя ответы бота
- **Безопасность**: токены хранятся только в `.env`, передаются через `Secret.from_token()` Haystack
