package app.devai;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.File;

/** Оболочка: интерфейс DevAI из assets, а модель работает на телефоне через встроенный движок llama.cpp. */
public class MainActivity extends Activity {
    private static final int PICK_MODEL = 41;

    private WebView web;
    private ModelManager models;
    private LlamaServer server;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        models = new ModelManager(this);
        server = new LlamaServer(this);
        models.setListener(file -> useAsync(file));

        web = new WebView(this);
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);   // история чата и настройки интерфейса
        web.addJavascriptInterface(new Bridge(), "DevAI");
        web.setWebViewClient(new WebViewClient());
        web.loadUrl("file:///android_asset/index.html");

        // Если модель уже скачана, запускаем последнюю использованную
        String last = models.lastUsed();
        if (!last.isEmpty() && models.fileFor(last).exists()) {
            useAsync(models.fileFor(last));
        } else if (!models.list().isEmpty()) {
            useAsync(models.list().get(0));
        }
    }

    private void useAsync(final File file) {
        models.message = "";
        models.setLastUsed(file.getName());
        new Thread(() -> server.start(file), "devai-start").start();
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == PICK_MODEL && res == RESULT_OK && data != null && data.getData() != null) {
            models.importFrom(data.getData());
        }
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (isFinishing()) server.stop();
        super.onDestroy();
    }

    private static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"') b.append("\\\"");
            else if (c == '\\') b.append("\\\\");
            else if (c == '\n') b.append("\\n");
            else if (c < 0x20) b.append(' ');
            else b.append(c);
        }
        return b.append('"').toString();
    }

    /** Методы, доступные интерфейсу как window.DevAI. */
    private class Bridge {
        @JavascriptInterface
        public String status() {
            String state = models.busy;
            if ("idle".equals(state)) state = server.state;
            String msg = !models.message.isEmpty() ? models.message : server.message;

            ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
            ((ActivityManager) getSystemService(Context.ACTIVITY_SERVICE)).getMemoryInfo(mi);

            StringBuilder sb = new StringBuilder("{");
            sb.append("\"state\":").append(q(state));
            sb.append(",\"message\":").append(q(msg));
            sb.append(",\"done\":").append(models.done);
            sb.append(",\"total\":").append(models.total);
            sb.append(",\"active\":").append(q(server.active));
            sb.append(",\"port\":").append(server.port);
            sb.append(",\"key\":").append(q(server.key));
            sb.append(",\"ramMb\":").append(mi.totalMem / 1048576L);
            sb.append(",\"freeMb\":").append(models.dir.getUsableSpace() / 1048576L);
            sb.append(",\"models\":[");
            boolean first = true;
            for (File f : models.list()) {
                if (!first) sb.append(',');
                first = false;
                sb.append("{\"file\":").append(q(f.getName())).append(",\"size\":").append(f.length()).append('}');
            }
            sb.append("],\"catalog\":[");
            first = true;
            for (String[] c : ModelManager.CATALOG) {
                if (!first) sb.append(',');
                first = false;
                String url = c[3];
                sb.append("{\"id\":").append(q(c[0])).append(",\"name\":").append(q(c[1]))
                  .append(",\"size\":").append(q(c[2]))
                  .append(",\"file\":").append(q(url.substring(url.lastIndexOf('/') + 1))).append('}');
            }
            sb.append("]}");
            return sb.toString();
        }

        @JavascriptInterface
        public void download(String id) { models.download(id); }

        @JavascriptInterface
        public void cancel() { models.cancel(); }

        @JavascriptInterface
        public void importModel() {
            runOnUiThread(() -> {
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                startActivityForResult(i, PICK_MODEL);
            });
        }

        @JavascriptInterface
        public void use(String file) {
            File f = models.fileFor(file);
            if (f.exists()) useAsync(f);
        }

        @JavascriptInterface
        public void remove(String file) {
            if (file.equals(server.active)) server.stop();
            models.remove(file);
        }
    }
}
