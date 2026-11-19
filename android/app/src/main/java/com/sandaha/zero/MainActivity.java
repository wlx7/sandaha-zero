package com.sandaha.zero;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Insets;
import android.os.Build;
import android.os.Bundle;
import android.view.DisplayCutout;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
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
import java.util.Locale;

public class MainActivity extends Activity {
    private WebView webView;
    private LinearLayout loadingOverlay;
    private long lastBackPress = 0;
    private int safeL = 0, safeT = 0, safeR = 0, safeB = 0;
    private volatile boolean pageLoaded = false;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            Window w = getWindow();
            if (w != null) {
                w.setStatusBarColor(Color.parseColor("#0a1913"));
                w.setNavigationBarColor(Color.parseColor("#0a1913"));
            }
        } catch (Throwable ignored) {
        }

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.parseColor("#0b1f18"));
        root.setKeepScreenOn(true);

        webView = new WebView(this);
        webView.setBackgroundColor(Color.parseColor("#0b1f18"));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setDatabaseEnabled(true);
        ws.setAllowFileAccess(false);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setUseWideViewPort(true);
        ws.setLoadWithOverviewMode(true);
        ws.setSupportZoom(false);
        ws.setBuiltInZoomControls(false);
        ws.setDisplayZoomControls(false);
        ws.setTextZoom(100);
        ws.setCacheMode(WebSettings.LOAD_NO_CACHE);

        webView.setWebChromeClient(new WebChromeClient());
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                pageLoaded = true;
                pushSafeInsetsToWeb();
                if (loadingOverlay != null) {
                    loadingOverlay.setVisibility(View.GONE);
                }
            }
        });

        root.setOnApplyWindowInsetsListener((v, insets) -> {
            try {
                if (insets == null) return insets;
                float density = Math.max(1f, getResources().getDisplayMetrics().density);
                int l = 0, t = 0, r = 0, b = 0;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    Insets vis = insets.getInsets(
                            WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                    l = Math.round(vis.left / density);
                    t = Math.round(vis.top / density);
                    r = Math.round(vis.right / density);
                    b = Math.round(vis.bottom / density);
                } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    DisplayCutout cutout = insets.getDisplayCutout();
                    if (cutout != null) {
                        l = Math.round(cutout.getSafeInsetLeft() / density);
                        t = Math.round(cutout.getSafeInsetTop() / density);
                        r = Math.round(cutout.getSafeInsetRight() / density);
                        b = Math.round(cutout.getSafeInsetBottom() / density);
                    }
                }
                if (l != safeL || t != safeT || r != safeR || b != safeB) {
                    safeL = l;
                    safeT = t;
                    safeR = r;
                    safeB = b;
                    if (pageLoaded) {
                        pushSafeInsetsToWeb();
                    }
                }
            } catch (Throwable ignored) {
            }
            return insets;
        });

        loadingOverlay = new LinearLayout(this);
        loadingOverlay.setOrientation(LinearLayout.VERTICAL);
        loadingOverlay.setGravity(Gravity.CENTER);
        loadingOverlay.setBackgroundColor(Color.parseColor("#0b1f18"));

        ProgressBar spinner = new ProgressBar(this);
        TextView title = new TextView(this);
        title.setText("三打哈 AI · SanDaHa-Zero");
        title.setTextColor(Color.parseColor("#e3b341"));
        title.setTextSize(20f);
        title.setGravity(Gravity.CENTER);
        title.setPadding(0, 32, 0, 10);

        final TextView subtitle = new TextView(this);
        subtitle.setText("正在启动本地 AI 引擎…");
        subtitle.setTextColor(Color.parseColor("#b7d2c6"));
        subtitle.setTextSize(14f);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setPadding(36, 0, 36, 0);

        loadingOverlay.addView(spinner);
        loadingOverlay.addView(title);
        loadingOverlay.addView(subtitle);

        root.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        root.addView(loadingOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(root);

        // Defer window insets / cutout configuration until DecorView is attached to WindowManager
        root.post(this::applyImmersiveFullscreen);

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
                runOnUiThread(() -> {
                    if (!isFinishing() && !isDestroyed() && webView != null) {
                        webView.loadUrl(url);
                    }
                });
            } catch (final Throwable e) {
                e.printStackTrace();
                final String msg = e.getClass().getSimpleName() + ": " + e.getMessage();
                runOnUiThread(() -> {
                    if (!isFinishing() && !isDestroyed()) {
                        subtitle.setText("启动失败: " + msg);
                        Toast.makeText(MainActivity.this, "AI 引擎启动异常: " + msg, Toast.LENGTH_LONG).show();
                    }
                });
            }
        }, "sandaha-boot").start();
    }

    @SuppressWarnings("deprecation")
    private void applyImmersiveFullscreen() {
        try {
            if (isFinishing() || isDestroyed()) return;
            Window w = getWindow();
            if (w == null) return;
            View decor = w.peekDecorView();
            if (decor == null || !decor.isAttachedToWindow()) return;

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                try {
                    WindowManager.LayoutParams lp = w.getAttributes();
                    if (lp.layoutInDisplayCutoutMode != WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS) {
                        lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
                        w.setAttributes(lp);
                    }
                } catch (Throwable ignored) {
                }
                w.setDecorFitsSystemWindows(false);
                WindowInsetsController ctrl = decor.getWindowInsetsController();
                if (ctrl != null) {
                    ctrl.hide(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                    ctrl.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
                }
            } else {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    try {
                        WindowManager.LayoutParams lp = w.getAttributes();
                        if (lp.layoutInDisplayCutoutMode != WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES) {
                            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
                            w.setAttributes(lp);
                        }
                    } catch (Throwable ignored) {
                    }
                }
                decor.setSystemUiVisibility(
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                );
            }
        } catch (Throwable ignored) {
        }
    }

    private void pushSafeInsetsToWeb() {
        try {
            if (!pageLoaded || webView == null || isFinishing() || isDestroyed()) return;
            final String js = String.format(Locale.US,
                    "(function(){var s=document.documentElement.style;" +
                    "s.setProperty('--safe-l','%dpx');s.setProperty('--safe-t','%dpx');" +
                    "s.setProperty('--safe-r','%dpx');s.setProperty('--safe-b','%dpx');" +
                    "window.dispatchEvent(new Event('resize'));})();",
                    safeL, safeT, safeR, safeB);
            webView.post(() -> {
                try {
                    if (webView != null && !isFinishing() && !isDestroyed()) {
                        webView.evaluateJavascript(js, null);
                    }
                } catch (Throwable ignored) {
                }
            });
        } catch (Throwable ignored) {
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            applyImmersiveFullscreen();
        }
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        if (webView != null) {
            webView.postDelayed(() -> {
                applyImmersiveFullscreen();
                pushSafeInsetsToWeb();
            }, 120);
        }
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
        try {
            if (pageLoaded && webView != null) {
                webView.evaluateJavascript("(window.__sdhBack ? window.__sdhBack() : false)", value -> {
                    if ("true".equals(value)) {
                        return;
                    }
                    long now = System.currentTimeMillis();
                    if (now - lastBackPress > 2000) {
                        lastBackPress = now;
                        Toast.makeText(MainActivity.this, "再按一次退出三打哈 AI", Toast.LENGTH_SHORT).show();
                    } else {
                        MainActivity.super.onBackPressed();
                    }
                });
                return;
            }
        } catch (Throwable ignored) {
        }
        super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        try {
            pageLoaded = false;
            if (webView != null) {
                webView.stopLoading();
                ViewParent parent = webView.getParent();
                if (parent instanceof ViewGroup) {
                    ((ViewGroup) parent).removeView(webView);
                }
                webView.destroy();
                webView = null;
            }
        } catch (Throwable ignored) {
        }
        super.onDestroy();
    }
}
