package app.devai;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Хранит модели .gguf в папке приложения, скачивает их и импортирует из файлов. */
class ModelManager {
    interface Listener { void onModelReady(File file); }

    private interface Work { File run() throws Exception; }

    /** Готовых загрузок в приложении нет: в комплекте только Socket 1.0, остальное пользователь добавляет своим файлом. */
    static final String[][] CATALOG = {};

    final File dir;
    volatile String busy = "idle";      // idle | downloading | importing
    volatile String message = "";
    volatile long done = 0;
    volatile long total = -1;
    volatile boolean cancel = false;

    private final Context ctx;
    private final SharedPreferences prefs;
    private Listener listener;

    ModelManager(Context context) {
        ctx = context.getApplicationContext();
        dir = new File(ctx.getFilesDir(), "models");
        dir.mkdirs();
        prefs = ctx.getSharedPreferences("devai", Context.MODE_PRIVATE);
    }

    void setListener(Listener l) { listener = l; }

    static final String BUNDLED = "socket-1.0.gguf";

    /** Кладёт встроенную модель Socket 1.0 в папку моделей (при первом запуске или по просьбе пользователя). */
    synchronized boolean installBundled(boolean force) {
        if (!force && prefs.getBoolean("bundled_done", false)) return false;
        File target = new File(dir, BUNDLED);
        try {
            if (!target.exists()) {
                File part = new File(dir, BUNDLED + ".part");
                try (InputStream in = ctx.getAssets().open(BUNDLED); OutputStream out = new FileOutputStream(part)) {
                    byte[] buf = new byte[64 * 1024];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                }
                if (!part.renameTo(target)) throw new IOException("не удалось сохранить файл");
            }
            prefs.edit().putBoolean("bundled_done", true).apply();
            return true;
        } catch (IOException e) {
            message = "Встроенной модели нет в этой сборке приложения";
            return false;
        }
    }

    boolean socketDefaultDone() { return prefs.getBoolean("socket_default_v1", false); }

    void markSocketDefault() { prefs.edit().putBoolean("socket_default_v1", true).apply(); }

    boolean isBusy() { return !"idle".equals(busy); }

    String lastUsed() { return prefs.getString("last", ""); }

    void setLastUsed(String name) { prefs.edit().putString("last", name).apply(); }

    List<File> list() {
        File[] files = dir.listFiles();
        List<File> out = new ArrayList<>();
        if (files != null) {
            Arrays.sort(files);
            for (File f : files) {
                if (f.isFile() && f.getName().endsWith(".gguf")) out.add(f);
            }
        }
        return out;
    }

    static String sanitize(String name) {
        String n = name == null ? "model" : name.replaceAll("[^A-Za-z0-9._-]", "_");
        if (n.startsWith(".")) n = "_" + n;
        if (!n.toLowerCase().endsWith(".gguf")) n = n + ".gguf";
        return n;
    }

    File fileFor(String name) { return new File(dir, sanitize(name)); }

    void remove(String name) {
        File f = fileFor(name);
        File part = new File(f.getPath() + ".part");
        f.delete();
        part.delete();
        if (f.getName().equals(lastUsed())) prefs.edit().remove("last").apply();
    }

    void cancel() { cancel = true; }

    /** Фоновая работа: скачивание или копирование. Прогресс виден в done/total. */
    private synchronized void runWorker(String kind, Work work) {
        if (isBusy()) return;
        busy = kind;
        message = "";
        cancel = false;
        done = 0;
        total = -1;
        new Thread(() -> {
            try {
                File f = work.run();
                busy = "idle";
                if (listener != null) listener.onModelReady(f);
            } catch (Exception e) {
                busy = "idle";
                message = cancel ? "Остановлено. Скачивание можно продолжить, оно начнётся с того же места."
                                 : "Ошибка: " + e.getMessage();
            }
        }, "devai-models").start();
    }

    void download(String id) {
        for (String[] c : CATALOG) {
            if (!c[0].equals(id)) continue;
            final String url = c[3];
            final String name = url.substring(url.lastIndexOf('/') + 1);
            runWorker("downloading", () -> downloadTo(url, new File(dir, name)));
            return;
        }
    }

    private File downloadTo(String urlStr, File target) throws IOException {
        File part = new File(target.getPath() + ".part");
        long have = part.exists() ? part.length() : 0;
        HttpURLConnection c = (HttpURLConnection) new URL(urlStr).openConnection();
        c.setConnectTimeout(20000);
        c.setReadTimeout(30000);
        c.setRequestProperty("User-Agent", "DevAI");
        if (have > 0) c.setRequestProperty("Range", "bytes=" + have + "-");
        int code = c.getResponseCode();
        if (code != 200 && code != 206) throw new IOException("сервер ответил " + code + ". Выберите свой файл .gguf кнопкой ниже");
        boolean resume = code == 206;
        if (!resume) have = 0;
        long len = c.getContentLengthLong();
        total = len > 0 ? len + have : -1;
        done = have;
        if (len > 0 && dir.getUsableSpace() < len + 100L * 1024 * 1024) {
            throw new IOException("не хватает места на телефоне");
        }
        try (InputStream in = c.getInputStream(); OutputStream out = new FileOutputStream(part, resume)) {
            copy(in, out);
        } finally {
            c.disconnect();
        }
        return finish(part, target);
    }

    void importFrom(Uri uri) {
        final String name = sanitize(displayName(uri));
        runWorker("importing", () -> {
            File target = new File(dir, name);
            File part = new File(dir, name + ".part");
            total = querySize(uri);
            if (total > 0 && dir.getUsableSpace() < total + 100L * 1024 * 1024) {
                throw new IOException("не хватает места на телефоне");
            }
            try (InputStream in = ctx.getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(part)) {
                if (in == null) throw new IOException("не удалось открыть файл");
                copy(in, out);
            } catch (IOException e) {
                part.delete();
                throw e;
            }
            return finish(part, target);
        });
    }

    private void copy(InputStream in, OutputStream out) throws IOException {
        byte[] buf = new byte[64 * 1024];
        int n;
        while ((n = in.read(buf)) > 0) {
            if (cancel) throw new IOException("остановлено");
            out.write(buf, 0, n);
            done += n;
        }
    }

    private File finish(File part, File target) throws IOException {
        if (!isGguf(part)) {
            part.delete();
            throw new IOException("это не файл GGUF");
        }
        if (target.exists()) target.delete();
        if (!part.renameTo(target)) throw new IOException("не удалось сохранить файл");
        return target;
    }

    static boolean isGguf(File f) {
        try (FileInputStream in = new FileInputStream(f)) {
            byte[] m = new byte[4];
            return in.read(m) == 4 && m[0] == 'G' && m[1] == 'G' && m[2] == 'U' && m[3] == 'F';
        } catch (IOException e) {
            return false;
        }
    }

    private String displayName(Uri uri) {
        try (Cursor c = ctx.getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (i >= 0 && c.getString(i) != null) return c.getString(i);
            }
        } catch (Exception ignored) { }
        return "model.gguf";
    }

    private long querySize(Uri uri) {
        try (Cursor c = ctx.getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int i = c.getColumnIndex(OpenableColumns.SIZE);
                if (i >= 0 && !c.isNull(i)) return c.getLong(i);
            }
        } catch (Exception ignored) { }
        return -1;
    }
}
