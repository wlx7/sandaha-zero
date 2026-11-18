package com.sandaha.zero;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.Window;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private WebView webView;
    private LinearLayout loadingOverlay;
    private long lastBackPress = 0;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setStatusBarColor(Color.parseColor("#0b241b"));
        getWindow().setNavigationBarColor(Color.parseColor("#0b241b"));

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.parseColor("#0b241b"));

        webView = new WebView(this);
        webView.setBackgroundColor(Color.parseColor("#0b241b"));
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setDatabaseEnabled(true);
        ws.setAllowFileAccess(false);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setUseWideViewPort(true);
        ws.setLoadWithOverviewMode(true);
        ws.setTextZoom(100);

        webView.setWebChromeClient(new WebChromeClient());
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                if (loadingOverlay != null) {
                    loadingOverlay.setVisibility( android.view.View.GONE );
                }
            }
        });

        loadingOverlay = new LinearLayout(this);
        loadingOverlay.setOrientation(LinearLayout.VERTICAL);
        loadingOverlay.setGravity(Gravity.CENTER);
        loadingOverlay.setBackgroundColor(Color.parseColor("#0b241b"));

        ProgressBar spinner = new ProgressBar(this);
        TextView title = new TextView(this);
        title.setText("三打哈 AI · SanDaHa-Zero");
        title.setTextColor(Color.parseColor("#e3b341"));
        title.setTextSize(20f);
        title.setGravity(Gravity.CENTER);
        title.setPadding(0, 36, 0, 12);

        TextView subtitle = new TextView(this);
        subtitle.setText("正在启动本地 AI 引擎…");
        subtitle.setTextColor(Color.parseColor("#b7d2c6"));
        subtitle.setTextSize(14f);
        subtitle.setGravity(Gravity.CENTER);

        loadingOverlay.addView(spinner);
        loadingOverlay.addView(title);
        loadingOverlay.addView(subtitle);

        root.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        root.addView(loadingOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(root);

        new Thread(() -> {
            try {
                File sandahaRoot = new File(getFilesDir(), "sandaha_root");
                syncAssetsIfNeeded(getAssets(), sandahaRoot);

                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(getApplicationContext()));
                }
                Python py = Python.getInstance();
                PyObject mod = py.getModule("sandaha.server.lite_server");
                int port = mod.callAttr("start_android_server", 8765, sandahaRoot.getAbsolutePath()).toInt();
                final String url = "http://127.0.0.1:" + port + "/";
                runOnUiThread(() -> webView.loadUrl(url));
            } catch (final Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    subtitle.setText("启动失败: " + e.getMessage());
                    Toast.makeText(MainActivity.this, "AI 引擎启动异常: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }, "sandaha-boot").start();
    }

    private void syncAssetsIfNeeded(AssetManager am, File destRoot) throws IOException {
        String bundledVer = readAssetText(am, "version.txt");
        File verFile = new File(destRoot, "version.txt");
        if (verFile.exists()) {
            String curVer = readFileText(verFile);
            if (bundledVer.equals(curVer) && new File(destRoot, "models/sandaha-best.npz").exists()) {
                return;
            }
        }
        copyAssetTree(am, "web", new File(destRoot, "web"));
        copyAssetTree(am, "models", new File(destRoot, "models"));
        destRoot.mkdirs();
        try (FileOutputStream fos = new FileOutputStream(verFile)) {
            fos.write(bundledVer.getBytes(StandardCharsets.UTF_8));
        }
    }

    private void copyAssetTree(AssetManager am, String assetPath, File dst) throws IOException {
        String[] children = am.list(assetPath);
        if (children != null && children.length > 0) {
            dst.mkdirs();
            for (String child : children) {
                copyAssetTree(am, assetPath + "/" + child, new File(dst, child));
            }
        } else {
            File parent = dst.getParentFile();
            if (parent != null) parent.mkdirs();
            try (InputStream in = am.open(assetPath);
                 FileOutputStream out = new FileOutputStream(dst)) {
                byte[] buf = new byte[32768];
                int n;
                while ((n = in.read(buf)) > 0) {
                    out.write(buf, 0, n);
                }
            }
        }
    }

    private String readAssetText(AssetManager am, String path) {
        try (InputStream in = am.open(path)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int n;
            while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
            return bos.toString("UTF-8").trim();
        } catch (IOException e) {
            return "";
        }
    }

    private String readFileText(File f) {
        try (FileInputStream in = new FileInputStream(f)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int n;
            while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
            return bos.toString("UTF-8").trim();
        } catch (IOException e) {
            return "";
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        if (webView != null) {
            String url = webView.getUrl();
            if (url != null && (url.contains("#/play") || url.contains("#/train"))) {
                webView.evaluateJavascript("location.hash = '#/';", null);
                return;
            }
            if (webView.canGoBack()) {
                webView.goBack();
                return;
            }
        }
        long now = System.currentTimeMillis();
        if (now - lastBackPress > 2000) {
            lastBackPress = now;
            Toast.makeText(this, "再按一次退出三打哈 AI", Toast.LENGTH_SHORT).show();
        } else {
            super.onBackPressed();
        }
    }
}
