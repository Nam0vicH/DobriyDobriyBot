"""
DobriyDobriyBot — Двухуровневая антиспам-система для Telegram.

Уровень 1: Быстрый RegEx-фильтр (patterns.json)
Уровень 2: AI-арбитр (Gemini LLM)
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── Конфигурация ────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

BASE_DIR = Path(__file__).resolve().parent
PATTERNS_FILE = BASE_DIR / "patterns.json"
PROMPT_FILE = BASE_DIR / "prompt.txt"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "spam.log"

# ─── Логирование ─────────────────────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)

# Корневой логгер бота
logger = logging.getLogger("antispam")
logger.setLevel(logging.INFO)

# Формат
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

# Консольный хендлер
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
logger.addHandler(console_handler)

# Файловый хендлер
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)


# ─── Загрузка данных ─────────────────────────────────────────────────────────

def load_patterns() -> dict:
    """Загрузка regex-паттернов из patterns.json."""
    with open(PATTERNS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "Загружены паттерны: %d валют, %d действий, %d исключений",
        len(data.get("currencies", [])),
        len(data.get("actions", [])),
        len(data.get("ignore_words", [])),
    )
    return data


def load_prompt() -> str:
    """Загрузка системного промта для AI-арбитра из prompt.txt."""
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    logger.info("Загружен промт AI-арбитра (%d символов)", len(text))
    return text


# ─── Уровень 1: Быстрый RegEx-фильтр ────────────────────────────────────────

def quick_regex_filter(text: str, patterns: dict) -> bool:
    """
    Уровень 1: быстрая проверка сообщения по regex-паттернам.

    Возвращает True, если сообщение ПОДОЗРИТЕЛЬНО
    (найдена валюта + действие), но не содержит ignore-слов.
    """
    lower_text = text.lower()

    # Если сообщение содержит ignore-слова — пропускаем сразу
    for word in patterns.get("ignore_words", []):
        if word in lower_text:
            return False

    # Нормализация: убираем пробелы между буквами/цифрами для обхода "u s d t"
    normalized = re.sub(r"(?<=\w)\s+(?=\w)", "", lower_text)

    has_currency = False
    has_action = False

    for currency in patterns.get("currencies", []):
        pattern = re.compile(re.escape(currency), re.IGNORECASE)
        if pattern.search(lower_text) or pattern.search(normalized):
            has_currency = True
            break

    for action in patterns.get("actions", []):
        pattern = re.compile(re.escape(action), re.IGNORECASE)
        if pattern.search(lower_text) or pattern.search(normalized):
            has_action = True
            break

    return has_currency and has_action


# ─── Уровень 2: AI-арбитр (Gemini) ──────────────────────────────────────────

def ai_arbiter(text: str, system_prompt: str, client: genai.Client) -> dict:
    """
    Уровень 2: отправляет подозрительное сообщение в Gemini для анализа.

    Возвращает словарь: {"is_spam": bool, "confidence": float, "reason": str}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Проанализируй это сообщение из Telegram-чата:\n\n\"{text}\"",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)
        logger.info(
            "AI-арбитр: is_spam=%s, confidence=%.2f, reason=%s",
            result.get("is_spam"),
            result.get("confidence", 0),
            result.get("reason", "—"),
        )
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error("Ошибка AI-арбитра: %s", e)
        # В случае ошибки считаем спамом (осторожный подход)
        return {
            "is_spam": True,
            "confidence": 0.0,
            "reason": f"Ошибка AI-арбитра: {e}",
        }


# ─── Формирование уведомления для админа ─────────────────────────────────────

def format_admin_alert(update: Update, ai_result: dict) -> str:
    """Формирует текст уведомления для администратора."""
    user = update.effective_user
    chat = update.effective_chat
    msg = update.effective_message

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "🚨 <b>СПАМ ОБНАРУЖЕН</b>",
        "",
        f"👤 <b>Пользователь:</b> {user.full_name}",
        f"🆔 <b>User ID:</b> <code>{user.id}</code>",
        f"📛 <b>Username:</b> @{user.username}" if user.username else "",
        f"💬 <b>Чат:</b> {chat.title or chat.id}",
        f"🕐 <b>Время:</b> {now}",
        "",
        f"📝 <b>Сообщение:</b>",
        f"<pre>{msg.text[:500]}</pre>",
        "",
        f"🤖 <b>AI-вердикт:</b>",
        f"   Спам: {'✅ Да' if ai_result.get('is_spam') else '❌ Нет'}",
        f"   Уверенность: {ai_result.get('confidence', 0):.0%}",
        f"   Причина: {ai_result.get('reason', '—')}",
    ]

    return "\n".join(line for line in lines if line is not None)


# ─── Telegram-хендлер ────────────────────────────────────────────────────────

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик входящих сообщений — двухуровневая фильтрация."""
    if not update.effective_message or not update.effective_message.text:
        return

    text = update.effective_message.text
    user = update.effective_user
    patterns = context.bot_data["patterns"]
    system_prompt = context.bot_data["system_prompt"]
    gemini_client = context.bot_data["gemini_client"]

    # ─── УРОВЕНЬ 1: Быстрый RegEx-фильтр ─────────────────────────────────
    if not quick_regex_filter(text, patterns):
        # Нет подозрений — пропускаем
        return

    logger.info(
        "⚠️  RegEx-фильтр сработал | user=%s (id=%s) | text=%s",
        user.full_name, user.id, text[:80],
    )

    # ─── УРОВЕНЬ 2: AI-арбитр ────────────────────────────────────────────
    ai_result = ai_arbiter(text, system_prompt, gemini_client)

    if not ai_result.get("is_spam", False):
        logger.info(
            "✅ AI-арбитр: НЕ спам | user=%s (id=%s) | reason=%s",
            user.full_name, user.id, ai_result.get("reason", "—"),
        )
        return

    # ─── СПАМ ПОДТВЕРЖДЁН — действуем ─────────────────────────────────────
    logger.warning(
        "🚫 СПАМ ПОДТВЕРЖДЁН | user=%s (id=%s) | confidence=%.2f | reason=%s",
        user.full_name, user.id,
        ai_result.get("confidence", 0),
        ai_result.get("reason", "—"),
    )

    # 1. Удаляем сообщение
    try:
        await update.effective_message.delete()
        logger.info("🗑️  Сообщение удалено")
    except Exception as e:
        logger.error("Не удалось удалить сообщение: %s", e)

    # 2. Баним пользователя
    try:
        await update.effective_chat.ban_member(user.id)
        logger.info("🔨 Пользователь забанен: %s (id=%s)", user.full_name, user.id)
    except Exception as e:
        logger.error("Не удалось забанить пользователя: %s", e)

    # 3. Уведомление админу
    if ADMIN_CHAT_ID:
        try:
            alert_text = format_admin_alert(update, ai_result)
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=alert_text,
                parse_mode="HTML",
            )
            logger.info("📩 Уведомление отправлено админу (chat_id=%s)", ADMIN_CHAT_ID)
        except Exception as e:
            logger.error("Не удалось отправить уведомление админу: %s", e)


# ─── Запуск бота ─────────────────────────────────────────────────────────────

def main() -> None:
    """Точка входа — запуск Telegram-бота."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY не задан в .env")

    # Загрузка данных
    patterns = load_patterns()
    system_prompt = load_prompt()

    # Инициализация Gemini-клиента
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    # Сборка Telegram-приложения
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Сохраняем данные в bot_data для доступа из хендлеров
    app.bot_data["patterns"] = patterns
    app.bot_data["system_prompt"] = system_prompt
    app.bot_data["gemini_client"] = gemini_client

    # Обрабатываем только текстовые сообщения в группах
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            on_message,
        )
    )

    logger.info("🤖 DobriyDobriyBot запущен!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
