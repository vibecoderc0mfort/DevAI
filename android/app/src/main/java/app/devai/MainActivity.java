package app.devai;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Insets;
import android.graphics.drawable.ColorDrawable;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.File;

/** Оболочка: интерфейс DevAI из assets, а модель работает на телефоне через встроенный движок llama.cpp. */
public class MainActivity extends Activity {
    private static final int PICK_MODEL = 41;

    private static final int DARK_BG = 0xFF050506;     // совпадает с фоном интерфейса, чтобы не было белых полос
    private static final int LIGHT_BG = 0xFFF3F4F8;

    private WebView web;
    private ModelManager models;
    private LlamaServer server;
    private int topInset = 0, bottomInset = 0;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        models = new ModelManager(this);
        server = new LlamaServer(this);
        models.setListener(file -> useAsync(file));

        boolean night = (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
        String theme = getSharedPreferences("devai", MODE_PRIVATE).getString("theme", night ? "dark" : "light");

        web = new WebView(this);
        web.setBackgroundColor("light".equals(theme) ? LIGHT_BG : DARK_BG);
        setContentView(web);
        setupEdgeToEdge();
        applyTheme(theme);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);   // история чатов и настройки интерфейса
        web.addJavascriptInterface(new Bridge(), "DevAI");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                pushInsets();
            }
        });
        web.loadUrl("file:///android_asset/index.html");

        // Первый запуск: ставим встроенную Socket 1.0, затем запускаем последнюю использованную модель
        new Thread(() -> {
            models.installBundled(false);
            if (!models.socketDefaultDone()) {            // после обновления по умолчанию запускаем Socket 1.0
                File socket = models.fileFor(ModelManager.BUNDLED);
                if (socket.exists()) models.setLastUsed(socket.getName());
                models.markSocketDefault();
            }
            String last = models.lastUsed();
            if (!last.isEmpty() && models.fileFor(last).exists()) {
                useAsync(models.fileFor(last));
            } else if (!models.list().isEmpty()) {
                useAsync(models.list().get(0));
            }
        }, "devai-init").start();
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

    /** Приложение рисуется под системными панелями: вместо белой полосы внизу виден фон приложения. */
    private void setupEdgeToEdge() {
        Window w = getWindow();
        w.setStatusBarColor(Color.TRANSPARENT);
        w.setNavigationBarColor(Color.TRANSPARENT);
        if (Build.VERSION.SDK_INT >= 29) w.setNavigationBarContrastEnforced(false);
        if (Build.VERSION.SDK_INT >= 28) {
            WindowManager.LayoutParams lp = w.getAttributes();
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            w.setAttributes(lp);
        }
        if (Build.VERSION.SDK_INT >= 30) {
            w.setDecorFitsSystemWindows(false);
        } else {
            w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
        }
        web.setOnApplyWindowInsetsListener((v, ins) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                Insets bars = ins.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                Insets ime = ins.getInsets(WindowInsets.Type.ime());
                topInset = bars.top;
                bottomInset = Math.max(bars.bottom, ime.bottom);
            } else {
                topInset = ins.getSystemWindowInsetTop();
                bottomInset = ins.getSystemWindowInsetBottom();
            }
            pushInsets();
            return ins;
        });
        web.requestApplyInsets();
    }

    /** Передаёт интерфейсу высоту системных панелей, чтобы кнопки не оказывались под ними. */
    private void pushInsets() {
        float d = getResources().getDisplayMetrics().density;
        web.evaluateJavascript("window.__setInsets&&window.__setInsets(" + (topInset / d) + "," + (bottomInset / d) + ")", null);
    }

    /** Цвет фона окна и значки системных панелей подстраиваются под выбранную тему. */
    private void applyTheme(String theme) {
        boolean dark = !"light".equals(theme);
        int bg = dark ? DARK_BG : LIGHT_BG;
        getSharedPreferences("devai", MODE_PRIVATE).edit().putString("theme", dark ? "dark" : "light").apply();
        getWindow().setBackgroundDrawable(new ColorDrawable(bg));
        if (web != null) web.setBackgroundColor(bg);
        if (Build.VERSION.SDK_INT >= 30) {
            WindowInsetsController c = getWindow().getInsetsController();
            if (c != null) {
                int mask = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS;
                c.setSystemBarsAppearance(dark ? 0 : mask, mask);
            }
        } else {
            View decor = getWindow().getDecorView();
            int f = decor.getSystemUiVisibility();
            int light = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
            decor.setSystemUiVisibility(dark ? (f & ~light) : (f | light));
        }
    }

    @Override
    public void onBackPressed() {
        // Сначала интерфейс закрывает свои панели (чаты, настройки), и только потом приложение закрывается
        web.evaluateJavascript("window.__back ? window.__back() : false", v -> {
            if (!"true".equals(v)) finish();
        });
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
        public void setTheme(String theme) {
            runOnUiThread(() -> applyTheme(theme));
        }

        @JavascriptInterface
        public void restoreBundled() {
            new Thread(() -> {
                models.message = "";
                if (models.installBundled(true)) useAsync(models.fileFor(ModelManager.BUNDLED));
            }).start();
        }

        @JavascriptInterface
        public void remove(String file) {
            if (file.equals(server.active)) server.stop();
            models.remove(file);
        }
    }
}
