package app.devai;

import android.content.Context;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.URL;
import java.security.SecureRandom;
import java.util.ArrayDeque;

/** Запускает движок llama.cpp (llama-server) как процесс внутри приложения и ждёт, пока модель загрузится. */
class LlamaServer {
    volatile String state = "idle";     // idle | starting | ready | error
    volatile String message = "";
    volatile String active = "";
    volatile int port = 0;
    volatile String key = "";

    private final Context ctx;
    private Process proc;
    private int generation = 0;
    private final ArrayDeque<String> tail = new ArrayDeque<>();

    LlamaServer(Context context) { ctx = context.getApplicationContext(); }

    synchronized void start(File model) {
        stopProcess();
        final int gen = ++generation;
        state = "starting";
        message = "";
        active = model.getName();
        key = randomKey();
        synchronized (tail) { tail.clear(); }

        String libDir = ctx.getApplicationInfo().nativeLibraryDir;
        File exe = new File(libDir, "libllama_server.so");
        if (!exe.exists()) {
            fail(gen, "В приложении нет движка модели. Установите apk, собранный через GitHub Actions целиком.");
            return;
        }
        try {
            port = freePort();
            ProcessBuilder pb = new ProcessBuilder(
                exe.getPath(),
                "-m", model.getPath(),
                "--host", "127.0.0.1",
                "--port", String.valueOf(port),
                "--api-key", key,
                "-c", "8192",
                "-t", String.valueOf(threads()),
                "-np", "1");
            pb.environment().put("LD_LIBRARY_PATH", libDir);
            pb.directory(new File(libDir));
            pb.redirectErrorStream(true);
            proc = pb.start();
        } catch (IOException e) {
            fail(gen, "Не удалось запустить движок: " + e.getMessage());
            return;
        }
        final Process p = proc;
        final int myPort = port;
        final String myKey = key;

        Thread reader = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    synchronized (tail) {
                        tail.addLast(line);
                        if (tail.size() > 25) tail.removeFirst();
                    }
                }
            } catch (IOException ignored) { }
        }, "devai-llama-log");
        reader.setDaemon(true);
        reader.start();

        Thread watcher = new Thread(() -> {
            long deadline = System.currentTimeMillis() + 5 * 60 * 1000;
            boolean ready = false;
            while (gen == generation && !ready && System.currentTimeMillis() < deadline) {
                if (!p.isAlive()) { crashed(gen); return; }
                ready = healthy(myPort, myKey);
                if (!ready) sleep(500);
            }
            if (gen != generation) return;
            if (!ready) { fail(gen, "Модель слишком долго загружается. Попробуйте модель поменьше."); return; }
            state = "ready";
            try { p.waitFor(); } catch (InterruptedException ignored) { }
            if (gen == generation) crashed(gen);
        }, "devai-llama-watch");
        watcher.setDaemon(true);
        watcher.start();
    }

    synchronized void stop() {
        generation++;
        stopProcess();
        state = "idle";
        message = "";
    }

    private void stopProcess() {
        if (proc != null) {
            proc.destroy();
            try { Thread.sleep(200); } catch (InterruptedException ignored) { }
            if (proc.isAlive()) proc.destroyForcibly();
            proc = null;
        }
    }

    private void crashed(int gen) {
        String log;
        synchronized (tail) { log = String.join("\n", tail); }
        String hint = "Модель остановилась. Скорее всего, не хватило памяти: выберите модель поменьше.";
        String last = log.isEmpty() ? "" : "\n\n" + lastLines(log, 6);
        fail(gen, hint + last);
    }

    private static String lastLines(String s, int n) {
        String[] lines = s.split("\n");
        StringBuilder b = new StringBuilder();
        for (int i = Math.max(0, lines.length - n); i < lines.length; i++) b.append(lines[i]).append('\n');
        return b.toString().trim();
    }

    private synchronized void fail(int gen, String msg) {
        if (gen != generation) return;
        state = "error";
        message = msg;
    }

    private static boolean healthy(int port, String key) {
        try {
            HttpURLConnection c = (HttpURLConnection) new URL("http://127.0.0.1:" + port + "/health").openConnection();
            c.setConnectTimeout(800);
            c.setReadTimeout(1500);
            c.setRequestProperty("Authorization", "Bearer " + key);
            int code = c.getResponseCode();
            c.disconnect();
            return code == 200;
        } catch (IOException e) {
            return false;
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) { }
    }

    private static int threads() {
        return Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));
    }

    private static int freePort() throws IOException {
        try (ServerSocket s = new ServerSocket(0, 0, InetAddress.getByName("127.0.0.1"))) {
            return s.getLocalPort();
        }
    }

    private static String randomKey() {
        byte[] b = new byte[16];
        new SecureRandom().nextBytes(b);
        StringBuilder sb = new StringBuilder();
        for (byte x : b) sb.append(String.format("%02x", x));
        return sb.toString();
    }
}
