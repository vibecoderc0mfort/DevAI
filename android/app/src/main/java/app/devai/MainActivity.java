package app.devai;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/** Оболочка: открывает DevAI, запущенный на вашем компьютере в той же сети. */
public class MainActivity extends Activity {
    private static final String CONNECT_PAGE = "file:///android_asset/connect.html";

    private WebView web;
    private SharedPreferences prefs;
    private String pendingMessage = "";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        prefs = getSharedPreferences("devai", MODE_PRIVATE);

        web = new WebView(this);
        setContentView(web);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);

        web.addJavascriptInterface(new Bridge(), "DevAIApp");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) {
                    showConnect("Не удалось подключиться. Проверьте адрес и что DevAI запущен на компьютере.");
                }
            }
        });

        String saved = prefs.getString("server", "");
        if (saved.isEmpty()) {
            showConnect("");
        } else {
            web.loadUrl(saved);
        }
    }

    private void showConnect(String message) {
        pendingMessage = message;
        web.loadUrl(CONNECT_PAGE);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    /** Методы, доступные странице как window.DevAIApp. */
    private class Bridge {
        @JavascriptInterface
        public String saved() {
            return prefs.getString("server", "");
        }

        @JavascriptInterface
        public String message() {
            return pendingMessage;
        }

        @JavascriptInterface
        public void connect(String address) {
            String url = address == null ? "" : address.trim();
            if (url.isEmpty()) return;
            if (!url.startsWith("http://") && !url.startsWith("https://")) {
                url = "http://" + url;
            }
            final String target = url;
            prefs.edit().putString("server", target).apply();
            runOnUiThread(() -> web.loadUrl(target));
        }

        @JavascriptInterface
        public void reset() {
            runOnUiThread(() -> showConnect(""));
        }
    }
}
