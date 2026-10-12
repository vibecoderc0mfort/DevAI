#!/usr/bin/env python3
"""DevAI 0.1 alpha — локальный ИИ-помощник для написания кода.

Работает поверх открытых моделей через Ollama, полностью на вашем компьютере.
Без интернета отвечает как обычно; с интернетом может дополнительно искать в сети.
Зависимости: только стандартная библиотека Python 3.8+.
"""
import argparse
import html
import json
import os
import re
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = "0.1 alpha"
# В собранном exe файлы лежат во временной папке PyInstaller (sys._MEIPASS)
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
OLLAMA = OLLAMA.replace("0.0.0.0", "127.0.0.1")

DEFAULT_MODEL = os.environ.get("DEVAI_MODEL", "qwen2.5-coder:7b")
NUM_CTX = int(os.environ.get("DEVAI_CTX", "8192"))  # размер контекста в токенах

SYSTEM_PROMPT = (
    "Ты DevAI — ИИ-помощник для разработки. Помогаешь писать, отлаживать, "
    "объяснять и улучшать код на любых языках и для любых платформ: приложения, "
    "сайты, боты, игры, скрипты, утилиты.\n"
    "Правила ответа:\n"
    "- Отвечай на языке пользователя.\n"
    "- Сначала коротко скажи, что собираешься сделать, затем давай код целиком.\n"
    "- Код всегда в блоках с указанием языка. Если проект из нескольких файлов, "
    "перед каждым блоком напиши путь файла.\n"
    "- Не сокращай код заглушками вроде «остальное без изменений».\n"
    "- Если требований не хватает, сделай разумные допущения и перечисли их.\n"
    "- В конце, если уместно, напиши, как запустить результат.\n"
    "- Если тебе передали результаты веб-поиска, опирайся на них и называй источники."
)


# --------------------------------------------------------------------------- сеть

def is_online(timeout=1.5):
    """Есть ли выход в интернет (без обращения к каким-либо сайтам)."""
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def lan_ip():
    """Адрес этого компьютера в локальной сети (пакеты не отправляются)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def ollama_get(path, timeout=3):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models():
    try:
        return [m["name"] for m in ollama_get("/api/tags").get("models", [])], True
    except Exception:
        return [], False


class OllamaError(Exception):
    def __init__(self, code, detail=""):
        super().__init__(code)
        self.code = code
        self.detail = detail


def ollama_chat(model, messages, system=SYSTEM_PROMPT):
    """Генератор: отдаёт разобранные JSON-куски ответа модели по мере генерации."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        "options": {"num_ctx": NUM_CTX},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise OllamaError("model_missing", model)
        raise OllamaError("ollama_error", e.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, ConnectionError, socket.timeout):
        raise OllamaError("ollama_down")
    with resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise OllamaError("ollama_error", str(obj["error"]))
            yield obj


def _clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def web_search(query, n=5):
    """Поиск через DuckDuckGo (HTML-версия). Нужен интернет."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DevAI"})
    with urllib.request.urlopen(req, timeout=8) as r:
        page = r.read().decode("utf-8", "replace")
    pattern = (
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>'
    )
    results = []
    for href, title, snippet in re.findall(pattern, page, re.S):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in q:
            href = q["uddg"][0]
        results.append({"title": _clean(title), "url": href, "snippet": _clean(snippet)})
        if len(results) >= n:
            break
    return results


# ------------------------------------------------------------------------- сервер

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (ROOT / "index.html").read_bytes())
        elif path == "/avatar.png":
            self._send(200, "image/png", (ROOT / "avatar.png").read_bytes())
        elif path == "/api/status":
            models, ollama_ok = list_models()
            data = {
                "version": VERSION,
                "online": is_online(),
                "ollama": ollama_ok,
                "models": models,
                "default": DEFAULT_MODEL,
            }
            self._send(200, "application/json", json.dumps(data).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/chat":
            return self._send(404, "text/plain; charset=utf-8", b"not found")
        try:
            size = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._send(400, "text/plain; charset=utf-8", b"bad request")
        self.stream_chat(payload)

    def stream_chat(self, payload):
        model = payload.get("model") or DEFAULT_MODEL
        messages = payload.get("messages") or []
        use_web = bool(payload.get("web"))
        system = SYSTEM_PROMPT

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        gen = None
        try:
            if use_web:
                if not is_online():
                    emit({"devai": "offline"})
                else:
                    last = next((m.get("content", "") for m in reversed(messages)
                                 if m.get("role") == "user"), "")
                    try:
                        found = web_search(last[:300])
                    except Exception:
                        found = []
                        emit({"devai": "search_failed"})
                    if found:
                        emit({"devai": "sources", "results": found})
                        system += "\n\nРезультаты веб-поиска по запросу пользователя:\n" + "\n".join(
                            f"[{i}] {r['title']} ({r['url']})\n{r['snippet']}"
                            for i, r in enumerate(found, 1)
                        )
            gen = ollama_chat(model, messages, system)
            for chunk in gen:
                emit(chunk)
        except OllamaError as e:
            try:
                emit({"error": e.code, "model": model, "detail": e.detail})
            except OSError:
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass  # пользователь нажал «Остановить» или закрыл вкладку
        finally:
            if gen is not None:
                gen.close()  # закрывает соединение с Ollama, генерация останавливается


class Server(ThreadingHTTPServer):
    daemon_threads = True


# --------------------------------------------------------------------------- CLI

def run_cli(model):
    print(f"DevAI {VERSION}, модель {model}. Команды: /new — новый чат, /exit — выход.")
    history = []
    while True:
        try:
            text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text == "/exit":
            return
        if text == "/new":
            history = []
            print("Контекст сброшен.")
            continue
        history.append({"role": "user", "content": text})
        print("DevAI: ", end="", flush=True)
        reply = ""
        try:
            for chunk in ollama_chat(model, history):
                piece = chunk.get("message", {}).get("content", "")
                reply += piece
                print(piece, end="", flush=True)
        except OllamaError as e:
            history.pop()
            if e.code == "ollama_down":
                print("\n[ошибка] Ollama не запущена. Установите её с ollama.com и запустите.")
            elif e.code == "model_missing":
                print(f"\n[ошибка] Модель не скачана. Выполните: ollama pull {model}")
            else:
                print(f"\n[ошибка] {e.detail}")
            continue
        history.append({"role": "assistant", "content": reply})
        print()


def main():
    global DEFAULT_MODEL
    ap = argparse.ArgumentParser(description=f"DevAI {VERSION}")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="модель Ollama по умолчанию")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="по умолчанию только этот компьютер; у интерфейса нет пароля")
    ap.add_argument("--cli", action="store_true", help="чат в терминале вместо браузера")
    ap.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = ap.parse_args()
    DEFAULT_MODEL = args.model

    if args.cli:
        return run_cli(args.model)

    url = f"http://{args.host}:{args.port}"
    server = Server((args.host, args.port), Handler)
    print(f"DevAI {VERSION} запущен: {url}  (модель {args.model}). Остановить: Ctrl+C")
    if args.host in ("0.0.0.0", "::"):
        ip = lan_ip()
        if ip:
            print(f"Для телефона (в той же Wi-Fi сети) введите адрес: {ip}:{args.port}")
        print("Внимание: у интерфейса нет пароля, он доступен всем в вашей сети.")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлен.")


if __name__ == "__main__":
    main()
