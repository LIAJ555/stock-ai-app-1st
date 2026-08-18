import os
import time
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google import genai

# ページ設定
st.set_page_config(page_title="日本株 AIチャート診断", layout="wide")

st.title("📈 日本株 AIチャート診断 & 分析ツール")

# 銘柄入力
col1, col2 = st.columns([3, 1])
with col1:
    code = st.text_input("銘柄コード（東証4桁）", value="9202")
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 診断実行", type="primary")

# APIキーの取得
api_key = st.secrets.get("GEMINI_API_KEY")

@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol):
    symbol = f"{ticker_symbol}.T"
    end_date = datetime.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=180)
    
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
        with st.spinner("株価データを取得・AI分析中..."):
            try:
                hist = get_stock_data(code)
                
                if hist.empty:
                    st.error(f"銘柄コード {code} のデータが見つかりませんでした。東証4桁コードを確認してください。")
                else:
                    # 移動平均線の計算
                    hist['SMA25'] = hist['Close'].rolling(window=25).mean()
                    hist['SMA75'] = hist['Close'].rolling(window=75).mean()

                    # チャート描画
                    fig, ax = plt.subplots(figsize=(10, 4.5))
                    ax.plot(hist.index, hist['Close'], label='Close', color='#1f77b4', linewidth=1.5)
                    ax.plot(hist.index, hist['SMA25'], label='25 SMA', color='#ff7f0e', linestyle='--', alpha=0.8)
                    ax.plot(hist.index, hist['SMA75'], label='75 SMA', color='#2ca02c', linestyle='--', alpha=0.8)
                    
                    ax.set_title(f"Stock Price Chart: {code}.T", fontsize=12)
                    ax.set_ylabel("Price (JPY)")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    ax.legend(loc='upper left')
                    fig.tight_layout()

                    # チャート表示
                    st.pyplot(fig)

                    # 直近データの抽出
                    latest_date = hist.index[-1].strftime('%Y-%m-%d')
                    latest_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    change = latest_price - prev_price
                    pct_change = (change / prev_price) * 100

                    st.metric(f"最新終値 / 現在値 ({latest_date} 時点)", f"¥{latest_price:,.1f}", f"{change:+,.1f} ({pct_change:+.2f}%)")

                    # Gemini による分析
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
あなたはプロの株式アナリストです。以下の東証銘柄データをもとに、テクニカル分析と投資判断のポイントを簡潔・論理的に解説してください。

【銘柄コード】: {code}
【基準日】: {latest_date}
【株価】: ¥{latest_price:,.1f} (前日比: {pct_change:+.2f}%)
【25日移動平均】: ¥{hist['SMA25'].iloc[-1]:,.1f}
【75日移動平均】: ¥{hist['SMA75'].iloc[-1]:,.1f}
【過去5日間の終値推移】: {list(hist['Close'].tail(5).round(1))}

以下の構成で回答してください：
1. **トレンド分析**（移動平均線との位置関係やモメンタム）
2. **注目すべきテクニカルポイント**（支持線・抵抗線、売買シグナル）
3. **投資スタンス・アドバイス**（短期・中長期それぞれの視点）
"""

                    # 安定モデルの指定
                    response = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt,
                    )

                    st.subheader("🤖 AIテクニカル診断レポート")
                    st.write(response.text)

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
