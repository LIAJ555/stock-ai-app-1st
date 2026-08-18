import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google import genai

# ページ基本設定
st.set_page_config(page_title="日本株 AIチャート診断", page_icon="📈", layout="wide")

st.title("📈 日本株 AIチャート診断 & 分析ツール")

# 1. APIキーの自動読み込み（StreamlitのSecretsから取得）
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ 設定")
    if not api_key:
        api_key = st.text_input("Gemini APIキー", type="password", placeholder="AIzaSy...")
        st.caption("Google AI StudioのAPIキーを入力してください。")
    else:
        st.success("✅ APIキー連携済み")
    st.markdown("---")
    st.caption("直近6ヶ月の株価チャート・テクニカル指標・Geminiプロ分析を即座に出力します。")

# 2. 銘柄入力エリア
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    code_input = st.text_input("銘柄コード（東証4桁）", value="9202", max_chars=4)
with col_in2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 診断実行", use_container_width=True, type="primary")

# 3. データ取得 & 指標計算
def get_stock_data(code):
    ticker = yf.Ticker(f"{code}.T")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if hist.empty or len(hist) < 25:
        return None, None
        
    hist.index = hist.index.tz_localize(None)
    hist["SMA25"] = hist["Close"].rolling(window=25).mean()
    hist["SMA75"] = hist["Close"].rolling(window=75).mean()
    
    delta = hist["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist["RSI"] = 100 - (100 / (1 + rs))
    #
    info = ticker.info
    raw_yield = info.get("dividendYield") or 0
    div_yield = raw_yield * 100 if 0 < raw_yield < 1 else raw_yield

    summary = {
        "name": info.get("longName") or info.get("shortName") or "不明",
        "sector": info.get("sector", "-"),
        "price": hist["Close"].iloc[-1],
        "change": ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100,
        "dev25": ((hist["Close"].iloc[-1] - hist["SMA25"].iloc[-1]) / hist["SMA25"].iloc[-1]) * 100,
        "rsi": hist["RSI"].iloc[-1] if not pd.isna(hist["RSI"].iloc[-1]) else 50.0,
        "pe": info.get("forwardPE", "-"),
        "pb": info.get("priceToBook", "-"),
        "yield": round(div_yield, 2) if div_yield else "-",
    }
    return hist, summary

# 4. 実行処理
if run_btn:
    code = code_input.strip()
    if not (code.isdigit() and len(code) == 4):
        st.error("4桁の半角数字（例: 9202）を入力してください。")
    else:
        with st.spinner(f"【{code}】のデータを取得中..."):
            hist, summary = get_stock_data(code)
            
        if hist is None:
            st.error(f"銘柄コード {code} のデータ取得に失敗しました。")
        else:
            st.subheader(f"🏢 {summary['name']} （コード: {code} ｜ {summary['sector']}）")
            
            # 指標タイル表示
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("現在株価", f"{summary['price']:,.1f} 円", f"{summary['change']:+.2f}%")
            m2.metric("予想PER", f"{summary['pe']} 倍" if summary['pe'] != '-' else '-')
            m3.metric("実績PBR", f"{summary['pb']} 倍" if summary['pb'] != '-' else '-')
            m4.metric("配当利回り", f"{summary['yield']} %" if summary['yield'] != '-' else '-')
            m5.metric("RSI (14日)", f"{summary['rsi']:.1f}")

            # チャート描画
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
            ax1.plot(hist.index, hist["Close"], label="終値", color="#1f77b4", linewidth=2)
            ax1.plot(hist.index, hist["SMA25"], label="25日線", color="#ff7f0e", linestyle="--")
            if not hist["SMA75"].dropna().empty:
                ax1.plot(hist.index, hist["SMA75"], label="75日線", color="#2ca02c", linestyle=":")
            ax1.set_ylabel("株価（円）")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left")
            
            avg_vol = hist["Volume"].iloc[-26:-1].mean()
            ax2.bar(hist.index, hist["Volume"], color="#aec7e8", width=0.7)
            ax2.axhline(y=avg_vol, color="red", linestyle=":", label="25日平均出来高")
            ax2.set_ylabel("出来高")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper left")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

            # Gemini プロ診断
            st.markdown("---")
            if not api_key:
                st.warning("⚠️ サイドバーにGemini APIキーを入力すると、ここにプロ診断コメントが表示されます。")
            else:
                with st.spinner("🤖 Geminiがプロ診断コメントを生成中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
プロの株式アナリストとして、以下の銘柄データから客観的かつ実践的な診断を行ってください。

【銘柄データ】
- 銘柄: 【{code}】{summary['name']} ({summary['sector']})
- 株価: {summary['price']:,.1f}円 (前日比: {summary['change']:+.2f}%)
- 25日線乖離率: {summary['dev25']:+.2f}% / RSI: {summary['rsi']:.1f}
- PER: {summary['pe']}倍 / PBR: {summary['pb']}倍 / 配当利回り: {summary['yield']}%

以下の構成で、簡潔・実践的に出力してください:
1. **総合判定（強気 / 中立 / 弱気 ＆ おすすめ度 ★1〜5）**
2. **テクニカル・需給評価**（過熱感・押し目水準・出来高の質・出尽くし警戒）
3. **ファンダメンタルズ評価**（割安度・PBR1倍割れ是正余地・還元姿勢）
4. **具体的な売買シナリオ**（エントリー目安株価・損切り/利確ライン）
"""
                        candidate_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
                        ai_text = None
                        for m in candidate_models:
                            try:
                                res = client.models.generate_content(model=m, contents=prompt)
                                if res and res.text:
                                    ai_text = res.text
                                    break
                            except Exception:
                                continue
                        
                        if ai_text:
                            st.markdown(f"### 🤖 Geminiのプロ分析 & 売買戦略\n\n{ai_text}")
                        else:
                            st.error("AI診断の生成に失敗しました。")
                    except Exception as e:
                        st.error(f"Gemini APIエラー: {e}")
