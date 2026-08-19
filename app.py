import os
import time
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google import genai

# ページ設定
st.set_page_config(page_title="日本株 AIフルテクニカル診断", layout="wide")

st.title("📈 日本株 AIフルテクニカル診断 & 分析ツール")

# 銘柄入力
col1, col2 = st.columns([3, 1])
with col1:
    code = st.text_input("銘柄コード（東証4桁）", value="9202")
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 フル診断実行", type="primary")

# APIキーの取得
api_key = st.secrets.get("GEMINI_API_KEY")

@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol):
    symbol = f"{ticker_symbol}.T"
    end_date = datetime.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=200)
    
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

if run_btn:
    if not code:
        st.warning("銘柄コードを入力してください。")
    elif not api_key:
        st.error("Gemini APIキーが設定されていません。StreamlitのSecretsを確認してください。")
    else:
        with st.spinner("株価データ取得・テクニカル指標計算・AI分析中..."):
            try:
                hist = get_stock_data(code)
                
                if hist.empty:
                    st.error(f"銘柄コード {code} のデータが見つかりませんでした。東証4桁コードを確認してください。")
                else:
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
                    ax1.set_title(f"Technical Analysis: {code}.T", fontsize=12)
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

                    # 最新数値の抽出
                    latest_date = hist.index[-1].strftime('%Y-%m-%d')
                    latest_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    change = latest_price - prev_price
                    pct_change = (change / prev_price) * 100

                    latest_vol = hist['Volume'].iloc[-1]
                    latest_rsi = hist['RSI'].iloc[-1]
                    latest_macd = hist['MACD'].iloc[-1]
                    latest_signal = hist['Signal'].iloc[-1]
                    latest_hist = hist['Hist'].iloc[-1]
                    bb_u = hist['BB_Upper'].iloc[-1]
                    bb_l = hist['BB_Lower'].iloc[-1]

                    st.metric(f"最新終値 / 現在値 ({latest_date} 時点)", f"¥{latest_price:,.1f}", f"{change:+,.1f} ({pct_change:+.2f}%)")

                    # Gemini によるフルテクニカル分析
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
あなたは百戦錬磨のプロのテクニカル株式アナリストです。
以下の東証銘柄データ（株価・移動平均線・ボリンジャーバンド・出来高・MACD・RSI）を統合的に分析し、投資判断とシナリオを分かりやすく解説してください。

【銘柄コード】: {code}
【基準日】: {latest_date}
【株価】: ¥{latest_price:,.1f} (前日比: {pct_change:+.2f}%)
【移動平均】: 25日SMA=¥{hist['SMA25'].iloc[-1]:,.1f} / 75日SMA=¥{hist['SMA75'].iloc[-1]:,.1f}
【ボリンジャーバンド(25日±2σ)】: 上限=¥{bb_u:,.1f} / 下限=¥{bb_l:,.1f}
【出来高】: {int(latest_vol):,} 株 (5日平均比: {latest_vol / hist['Vol_SMA5'].iloc[-1]:.2f}倍)
【MACD (12,26,9)】: MACD={latest_macd:.2f} / Signal={latest_signal:.2f} / Hist={latest_hist:.2f}
【RSI (14日)】: {latest_rsi:.1f}%
【過去5日間の終値】: {list(hist['Close'].tail(5).round(1))}

以下の項目に沿ってレポートを作成してください：
1. **トレンド & チャート形状**（移動平均の並び・傾き、ボリンジャーバンドのバンド幅と位置）
2. **オシレーター & モメンタム診断**（RSIの過熱感、MACDのシグナル・モメンタム）
3. **出来高の評価**（商いの増減とトレンドの整合性）
4. **テクニカル総合評価 & 投資戦略**（上値・下値のメド、短期・中長期それぞれの売買スタンス）
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

                    st.subheader("🤖 AIフルテクニカル診断レポート")
                    st.write(response_text)

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
