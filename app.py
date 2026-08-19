import os
import time
import urllib.parse
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import feedparser
from datetime import datetime, timedelta
from google import genai

# ページ設定
st.set_page_config(page_title="日本株 AI統合分析（テクニカル × ファンダ × ニュース）", layout="wide")

st.title("📈 日本株 AI統合分析 & 投資診断ツール")

# 銘柄入力
col1, col2 = st.columns([3, 1])
with col1:
    code = st.text_input("銘柄コード（東証4桁）", value="9202")
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 総合診断実行", type="primary")

# APIキーの取得
api_key = st.secrets.get("GEMINI_API_KEY")

@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol):
    symbol = f"{ticker_symbol}.T"
    ticker = yf.Ticker(symbol)
    
    # 過去200日の日足株価
    end_date = datetime.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=200)
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 企業基本情報
    info = ticker.info if hasattr(ticker, "info") else {}
    return df, info

@st.cache_data(ttl=300)
def get_company_news(company_name, ticker_code):
    """Google News RSS から銘柄に関連するニュース見出しを取得"""
    query = f"{company_name} {ticker_code} 株"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries[:5]:  # 直近5件
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "")
        })
    return news_items

if run_btn:
    if not code:
        st.warning("銘柄コードを入力してください。")
    elif not api_key:
        st.error("Gemini APIキーが設定されていません。StreamlitのSecretsを確認してください。")
    else:
        with st.spinner("株価・財務データ取得・ニュース収集・AI分析中..."):
            try:
                hist, info = get_stock_data(code)
                
                if hist.empty:
                    st.error(f"銘柄コード {code} のデータが見つかりませんでした。東証4桁コードを確認してください。")
                else:
                    # 企業名の抽出
                    company_name = info.get('shortName', info.get('longName', f'コード {code}'))
                    
                    # 1. 移動平均線 (SMA)
                    hist['SMA25'] = hist['Close'].rolling(window=25).mean()
                    hist['SMA75'] = hist['Close'].rolling(window=75).mean()

                    # 2. ボリンジャーバンド (25日, ±2σ)
                    sma25_std = hist['Close'].rolling(window=25).std()
                    hist['BB_Upper'] = hist['SMA25'] + (sma25_std * 2)
                    hist['BB_Lower'] = hist['SMA25'] - (sma25_std * 2)

                    # 3. MACD (短期12, 長期26, シグナル9)
                    exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
                    exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
                    hist['MACD'] = exp12 - exp26
                    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
                    hist['Hist'] = hist['MACD'] - hist['Signal']

                    # 4. RSI (14日)
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist['RSI'] = 100 - (100 / (1 + rs))

                    # 出来高の5日移動平均
                    hist['Vol_SMA5'] = hist['Volume'].rolling(window=5).mean()

                    # ニュース取得
                    news_list = get_company_news(company_name, code)

                    # 画面表示: 企業名と主要指標カード
                    st.subheader(f"🏢 {company_name} ({code}.T)")
                    
                    latest_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    change = latest_price - prev_price
                    pct_change = (change / prev_price) * 100
                    latest_date = hist.index[-1].strftime('%Y-%m-%d')

                    # 財務指標の整理
                    pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
                    pb_ratio = info.get('priceToBook', None)
                    div_yield = info.get('dividendYield', None)
                    div_yield_str = f"{div_yield * 100:.2f}%" if div_yield else "N/A"
                    mkt_cap = info.get('marketCap', None)
                    mkt_cap_str = f"¥{mkt_cap / 1_000_000_000_000:.2f} 兆" if mkt_cap and mkt_cap >= 1_000_000_000_000 else (f"¥{mkt_cap / 100_000_000:.0f} 億" if mkt_cap else "N/A")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("現在値", f"¥{latest_price:,.1f}", f"{change:+,.1f} ({pct_change:+.2f}%)")
                    m2.metric("PER (実績/予想)", f"{pe_ratio:.1f} 倍" if pe_ratio else "N/A")
                    m3.metric("PBR", f"{pb_ratio:.2f} 倍" if pb_ratio else "N/A")
                    m4.metric("配当利回り", div_yield_str)
                    m5.metric("時価総額", mkt_cap_str)

                    # チャート描画 (上下4段)
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                        4, 1, figsize=(10, 10), 
                        gridspec_kw={'height_ratios': [3, 1, 1.2, 1.2]}, 
                        sharex=True
                    )

                    # --- 段1: 株価・移動平均・ボリンジャーバンド ---
                    ax1.plot(hist.index, hist['Close'], label='Close', color='#1f77b4', linewidth=1.5)
                    ax1.plot(hist.index, hist['SMA25'], label='25 SMA', color='#ff7f0e', linestyle='--', alpha=0.8)
                    ax1.plot(hist.index, hist['SMA75'], label='75 SMA', color='#2ca02c', linestyle='--', alpha=0.8)
                    ax1.plot(hist.index, hist['BB_Upper'], label='BB +2σ', color='#9467bd', linestyle=':', alpha=0.7)
                    ax1.plot(hist.index, hist['BB_Lower'], label='BB -2σ', color='#9467bd', linestyle=':', alpha=0.7)
                    ax1.fill_between(hist.index, hist['BB_Lower'], hist['BB_Upper'], color='#9467bd', alpha=0.08)
                    ax1.set_title(f"Technical Chart: {company_name} ({code}.T)", fontsize=12)
                    ax1.set_ylabel("Price (JPY)")
                    ax1.grid(True, linestyle=":", alpha=0.6)
                    ax1.legend(loc='upper left', fontsize=8)

                    # --- 段2: 出来高 ---
                    ax2.bar(hist.index, hist['Volume'], color='#7f7f7f', alpha=0.5, label='Volume')
                    ax2.plot(hist.index, hist['Vol_SMA5'], color='#d62728', linewidth=1, label='Vol 5-SMA')
                    ax2.set_ylabel("Volume")
                    ax2.grid(True, linestyle=":", alpha=0.6)
                    ax2.legend(loc='upper left', fontsize=8)

                    # --- 段3: MACD ---
                    ax3.plot(hist.index, hist['MACD'], label='MACD', color='#1f77b4', linewidth=1.2)
                    ax3.plot(hist.index, hist['Signal'], label='Signal', color='#d62728', linestyle='--', linewidth=1.2)
                    colors = ['#ff7f0e' if val >= 0 else '#1f77b4' for val in hist['Hist']]
                    ax3.bar(hist.index, hist['Hist'], color=colors, alpha=0.4, width=0.8, label='Hist')
                    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
                    ax3.set_ylabel("MACD")
                    ax3.grid(True, linestyle=":", alpha=0.6)
                    ax3.legend(loc='upper left', fontsize=8)

                    # --- 段4: RSI ---
                    ax4.plot(hist.index, hist['RSI'], label='RSI (14)', color='#e377c2', linewidth=1.2)
                    ax4.axhline(70, color='red', linestyle='--', alpha=0.6)
                    ax4.axhline(30, color='blue', linestyle='--', alpha=0.6)
                    ax4.set_ylim(0, 100)
                    ax4.set_ylabel("RSI")
                    ax4.grid(True, linestyle=":", alpha=0.6)
                    ax4.legend(loc='upper left', fontsize=8)

                    fig.tight_layout()
                    st.pyplot(fig)

                    # ニュースのアコーディオン表示
                    with st.expander("📰 直近の関連ニュース・ヘッドライン（クリックで展開）", expanded=False):
                        if news_list:
                            for n in news_list:
                                st.markdown(f"- [{n['title']}]({n['link']})")
                        else:
                            st.write("直近のヘッドラインニュースは見つかりませんでした。")

                    # 最新数値の抽出
                    latest_vol = hist['Volume'].iloc[-1]
                    latest_rsi = hist['RSI'].iloc[-1]
                    latest_macd = hist['MACD'].iloc[-1]
                    latest_signal = hist['Signal'].iloc[-1]
                    latest_hist = hist['Hist'].iloc[-1]
                    bb_u = hist['BB_Upper'].iloc[-1]
                    bb_l = hist['BB_Lower'].iloc[-1]

                    # ニュースタイトル一覧文字列
                    news_titles_str = "\n".join([f"- {n['title']}" for n in news_list]) if news_list else "特になし"

                    # Gemini によるフル統合分析
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
あなたは百戦錬磨のシニア株式ストラテジストです。
以下の【企業基本・財務データ】【テクニカル指標】【直近ニュース】を多角的に統合分析し、プロの投資判断レポートを作成してください。

【対象企業】: {company_name} (コード: {code}.T)
【基準日】: {latest_date}
【現在株価】: ¥{latest_price:,.1f} (前日比: {pct_change:+.2f}%)

【ファンダメンタルズ & バリュエーション】
- PER: {pe_ratio:.1f}倍 (※未取得の場合はN/A)
- PBR: {pb_ratio:.2f}倍 (※未取得の場合はN/A)
- 配当利回り: {div_yield_str}
- 時価総額: {mkt_cap_str}

【テクニカル指標】
- 移動平均: 25日SMA=¥{hist['SMA25'].iloc[-1]:,.1f} / 75日SMA=¥{hist['SMA75'].iloc[-1]:,.1f}
- ボリンジャーバンド(25日±2σ): 上限=¥{bb_u:,.1f} / 下限=¥{bb_l:,.1f}
- 出来高: {int(latest_vol):,} 株 (5日平均比: {latest_vol / hist['Vol_SMA5'].iloc[-1]:.2f}倍)
- MACD (12,26,9): MACD={latest_macd:.2f} / Signal={latest_signal:.2f} / Hist={latest_hist:.2f}
- RSI (14日): {latest_rsi:.1f}%

【直近の関連ニュース見出し】
{news_titles_str}

---
以下の構成で分かりやすく、メリハリのある投資レポートを作成してください：
1. **総合診断サマリー**（現在の株価位置付けを一言で言うと？）
2. **テクニカル分析**（トレンドの方向性、オシレーターの過熱感、出来高の裏付け）
3. **ファンダメンタルズ & バリュエーション評価**（PER/PBRや配当利回りから見た割安度・魅力度）
4. **ニュース・外部要因の影響**（直近の材料や市場のテーマ性）
5. **投資戦略シナリオ**（エントリーポイント、ターゲット上値、損切り/サポートラインの目安）
"""

                    # API呼び出し
                    response_text = None
                    for attempt in range(3):
                        try:
                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=prompt,
                            )
                            response_text = response.text
                            break
                        except Exception as req_err:
                            if "503" in str(req_err) and attempt < 2:
                                time.sleep(2)
                                continue
                            raise req_err

                    st.subheader("🤖 AI総合診断レポート（テクニカル × ファンダ × ニュース）")
                    st.write(response_text)

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
