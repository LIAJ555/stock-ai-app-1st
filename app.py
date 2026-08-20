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
st.set_page_config(page_title="日本株 AI統合分析 & スクリーニング", layout="wide")

st.title("📈 日本株 AI統合分析 & 投資診断プラットフォーム")

# APIキーの取得
api_key = st.secrets.get("GEMINI_API_KEY")

# ==========================================
# 共通関数・データ取得
# ==========================================
@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol):
    symbol = f"{ticker_symbol}.T"
    ticker = yf.Ticker(symbol)
    
    end_date = datetime.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=200)
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    info = ticker.info if hasattr(ticker, "info") else {}
    return df, info

@st.cache_data(ttl=300)
def get_company_news(company_name, ticker_code):
    query = f"{company_name} {ticker_code} 株"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries[:5]:
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "")
        })
    return news_items

# TOPIX 100 銘柄リスト
TOPIX100_SYMBOLS = [
    "1925.T", "1928.T", "8801.T", "8802.T", "8830.T",
    "2502.T", "2503.T", "2802.T", "2914.T",
    "3382.T", "8267.T", "9983.T",
    "3407.T", "4063.T", "4188.T", "4452.T", "4901.T", "5020.T", "5108.T", "5401.T", "5802.T", "5803.T",
    "4502.T", "4503.T", "4507.T", "4519.T", "4543.T", "4568.T", "4578.T",
    "4307.T", "4661.T", "4689.T", "6098.T", "7832.T", "7974.T", "9735.T",
    "6146.T", "6201.T", "6273.T", "6301.T", "6326.T", "6367.T", "7011.T",
    "6501.T", "6503.T", "6701.T", "6702.T", "6723.T", "6752.T", "6758.T",
    "6762.T", "6857.T", "6861.T", "6902.T", "6920.T", "6954.T", "6971.T",
    "6981.T", "7733.T", "7741.T", "7751.T", "8035.T",
    "7203.T", "7267.T", "7269.T", "7270.T", "7309.T", "7936.T",
    "6178.T", "7182.T", "8306.T", "8308.T", "8309.T", "8316.T", "8411.T",
    "8591.T", "8604.T", "8630.T", "8697.T", "8725.T", "8750.T", "8766.T",
    "8001.T", "8002.T", "8015.T", "8031.T", "8053.T", "8058.T", "8113.T",
    "9020.T", "9021.T", "9022.T", "9101.T", "9104.T", "9202.T",
    "9432.T", "9433.T", "9434.T", "9984.T"
]

# ==========================================
# タブ切り替えレイアウト
# ==========================================
tab1, tab2 = st.tabs(["🔍 個別銘柄 精密診断", "🏆 TOPIX100 買い推奨スクリーニング"])

# ----------------------------------------------------
# TAB 1: 個別銘柄の精密診断
# ----------------------------------------------------
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        code = st.text_input("銘柄コード（東証4桁）", value="9202", key="individual_code")
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 総合診断実行", type="primary", key="individual_btn")

    if run_btn:
        if not code:
            st.warning("銘柄コードを入力してください。")
        elif not api_key:
            st.error("Gemini APIキーが設定されていません。StreamlitのSecretsを確認してください。")
        else:
            with st.spinner("株価・コンセンサス・財務データ取得・ニュース収集・AI分析中..."):
                try:
                    hist, info = get_stock_data(code)
                    
                    if hist.empty:
                        st.error(f"銘柄コード {code} のデータが見つかりませんでした。東証4桁コードを確認してください。")
                    else:
                        company_name = info.get('shortName', info.get('longName', f'コード {code}'))
                        
                        # テクニカル計算
                        hist['SMA25'] = hist['Close'].rolling(window=25).mean()
                        hist['SMA75'] = hist['Close'].rolling(window=75).mean()
                        sma25_std = hist['Close'].rolling(window=25).std()
                        hist['BB_Upper'] = hist['SMA25'] + (sma25_std * 2)
                        hist['BB_Lower'] = hist['SMA25'] - (sma25_std * 2)

                        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
                        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
                        hist['MACD'] = exp12 - exp26
                        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
                        hist['Hist'] = hist['MACD'] - hist['Signal']

                        delta = hist['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        hist['RSI'] = 100 - (100 / (1 + rs))

                        hist['Vol_SMA5'] = hist['Volume'].rolling(window=5).mean()
                        news_list = get_company_news(company_name, code)

                        latest_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                        change = latest_price - prev_price
                        pct_change = (change / prev_price) * 100
                        latest_date = hist.index[-1].strftime('%Y-%m-%d')

                        pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
                        book_value = info.get('bookValue', None)
                        raw_pbr = info.get('priceToBook', None)
                        pbr_val = (latest_price / book_value) if (book_value and book_value > 0 and latest_price > 0) else raw_pbr

                        mkt_cap = info.get('marketCap', None)
                        mkt_cap_str = f"¥{mkt_cap / 1_000_000_000_000:.2f} 兆" if mkt_cap and mkt_cap >= 1_000_000_000_000 else (f"¥{mkt_cap / 100_000_000:.0f} 億" if mkt_cap else "N/A")

                        div_rate = info.get('dividendRate', None)
                        raw_div_yield = info.get('dividendYield', None)
                        if div_rate and latest_price > 0:
                            calc_yield = (div_rate / latest_price) * 100
                            div_yield_str = f"{calc_yield:.2f}% (¥{div_rate:.0f})"
                        elif raw_div_yield is not None:
                            actual_yield = raw_div_yield if raw_div_yield > 1.0 else raw_div_yield * 100
                            div_yield_str = f"{actual_yield:.2f}%"
                        else:
                            div_yield_str = "N/A"

                        target_price = info.get('targetMeanPrice', None)
                        target_high = info.get('targetHighPrice', None)
                        target_low = info.get('targetLowPrice', None)
                        num_analysts = info.get('numberOfAnalystOpinions', None)
                        recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()

                        if target_price and latest_price > 0:
                            upside_pct = ((target_price - latest_price) / latest_price) * 100
                            target_str = f"¥{target_price:,.0f}"
                            target_delta = f"{upside_pct:+.1f}% (乖離率)"
                        else:
                            target_str = "N/A"
                            target_delta = None

                        # 指標カード
                        st.subheader(f"🏢 {company_name} ({code}.T)")
                        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
                        row1_col1.metric("現在値", f"¥{latest_price:,.1f}", f"{change:+,.1f} ({pct_change:+.2f}%)")
                        row1_col2.metric("アナリスト目標株価(平均)", target_str, target_delta)
                        row1_col3.metric("コンセンサス判断", recommendation, f"{num_analysts}名のアナリスト" if num_analysts else None)
                        row1_col4.metric("時価総額", mkt_cap_str)

                        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
                        row2_col1.metric("PER (実績/予想)", f"{pe_ratio:.1f} 倍" if pe_ratio else "N/A")
                        row2_col2.metric("PBR", f"{pbr_val:.2f} 倍" if pbr_val else "N/A")
                        row2_col3.metric("配当利回り (年間配当)", div_yield_str)
                        row2_col4.metric("52週レンジ", f"¥{info.get('fiftyTwoWeekLow', 0):,.0f} - ¥{info.get('fiftyTwoWeekHigh', 0):,.0f}" if info.get('fiftyTwoWeekHigh') else "N/A")

                        # チャート描画 (上下5段)
                        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                            5, 1, figsize=(10, 13), 
                            gridspec_kw={'height_ratios': [2.5, 2.5, 1.0, 1.2, 1.2]}, 
                            sharex=True
                        )

                        # 段1: 株価 & 移動平均 & 目標株価
                        ax1.plot(hist.index, hist['Close'], label='Close', color='#1f77b4', linewidth=1.5)
                        ax1.plot(hist.index, hist['SMA25'], label='25 SMA', color='#ff7f0e', linestyle='--', alpha=0.8)
                        ax1.plot(hist.index, hist['SMA75'], label='75 SMA', color='#2ca02c', linestyle='--', alpha=0.8)
                        if target_price:
                            ax1.axhline(target_price, color='#d62728', linestyle='-.', alpha=0.8, label=f'Target (¥{target_price:,.0f})')
                        ax1.set_title(f"Technical & Fundamental Charts: {company_name} ({code}.T)", fontsize=12)
                        ax1.set_ylabel("Price & SMA")
                        ax1.grid(True, linestyle=":", alpha=0.6)
                        ax1.legend(loc='upper left', fontsize=8)

                        # 段2: ボリンジャーバンド
                        ax2.plot(hist.index, hist['Close'], label='Close', color='#1f77b4', linewidth=1.2)
                        ax2.plot(hist.index, hist['SMA25'], label='25 SMA (Mid)', color='#ff7f0e', linestyle='--', alpha=0.6)
                        ax2.plot(hist.index, hist['BB_Upper'], label='BB +2σ', color='#9467bd', linestyle=':', alpha=0.8)
                        ax2.plot(hist.index, hist['BB_Lower'], label='BB -2σ', color='#9467bd', linestyle=':', alpha=0.8)
                        ax2.fill_between(hist.index, hist['BB_Lower'], hist['BB_Upper'], color='#9467bd', alpha=0.12)
                        ax2.set_ylabel("Bollinger Bands")
                        ax2.grid(True, linestyle=":", alpha=0.6)
                        ax2.legend(loc='upper left', fontsize=8)

                        # 段3: 出来高
                        ax3.bar(hist.index, hist['Volume'], color='#7f7f7f', alpha=0.5, label='Volume')
                        ax3.plot(hist.index, hist['Vol_SMA5'], color='#d62728', linewidth=1, label='Vol 5-SMA')
                        ax3.set_ylabel("Volume")
                        ax3.grid(True, linestyle=":", alpha=0.6)
                        ax3.legend(loc='upper left', fontsize=8)

                        # 段4: MACD
                        ax4.plot(hist.index, hist['MACD'], label='MACD', color='#1f77b4', linewidth=1.2)
                        ax4.plot(hist.index, hist['Signal'], label='Signal', color='#d62728', linestyle='--', linewidth=1.2)
                        colors = ['#ff7f0e' if val >= 0 else '#1f77b4' for val in hist['Hist']]
                        ax4.bar(hist.index, hist['Hist'], color=colors, alpha=0.4, width=0.8, label='Hist')
                        ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
                        ax4.set_ylabel("MACD")
                        ax4.grid(True, linestyle=":", alpha=0.6)
                        ax4.legend(loc='upper left', fontsize=8)

                        # 段5: RSI
                        ax5.plot(hist.index, hist['RSI'], label='RSI (14)', color='#e377c2', linewidth=1.2)
                        ax5.axhline(70, color='red', linestyle='--', alpha=0.6)
                        ax5.axhline(30, color='blue', linestyle='--', alpha=0.6)
                        ax5.set_ylim(0, 100)
                        ax5.set_ylabel("RSI")
                        ax5.grid(True, linestyle=":", alpha=0.6)
                        ax5.legend(loc='upper left', fontsize=8)

                        fig.tight_layout()
                        st.pyplot(fig)

                        # ニュースアコーディオン
                        with st.expander("📰 直近の関連ニュース・ヘッドライン（クリックで展開）", expanded=False):
                            if news_list:
                                for n in news_list:
                                    st.markdown(f"- [{n['title']}]({n['link']})")
                            else:
                                st.write("直近のヘッドラインニュースは見つかりませんでした。")

                        # AI統合分析
                        latest_vol = hist['Volume'].iloc[-1]
                        latest_rsi = hist['RSI'].iloc[-1]
                        latest_macd = hist['MACD'].iloc[-1]
                        latest_signal = hist['Signal'].iloc[-1]
                        latest_hist = hist['Hist'].iloc[-1]
                        bb_u = hist['BB_Upper'].iloc[-1]
                        bb_l = hist['BB_Lower'].iloc[-1]
                        news_titles_str = "\n".join([f"- {n['title']}" for n in news_list]) if news_list else "特になし"
                        target_info_text = f"¥{target_price:,.0f} (現在値からの乖離率: {upside_pct:+.1f}%) [レンジ: ¥{target_low:,.0f}〜¥{target_high:,.0f} / カバー数: {num_analysts}名 / 判断: {recommendation}]" if target_price else "データなし"
                        pbr_text = f"{pbr_val:.2f}倍" if pbr_val else "N/A"

                        client = genai.Client(api_key=api_key)
                        prompt = f"""
あなたは百戦錬磨のシニア株式ストラテジストです。
以下の【企業基本・財務データ】【アナリストコンセンサス】【テクニカル指標】【直近ニュース】を多角的に統合分析し、プロの投資判断レポートを作成してください。

【対象企業】: {company_name} (コード: {code}.T)
【基準日】: {latest_date}
【現在株価】: ¥{latest_price:,.1f} (前日比: {pct_change:+.2f}%)

【アナリストコンセンサス & 目標株価】
- コンセンサス目標株価: {target_info_text}

【ファンダメンタルズ & バリュエーション】
- PER: {pe_ratio:.1f}倍 (※未取得の場合はN/A)
- PBR: {pbr_text}
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
1. **総合診断サマリー**
2. **アナリストコンセンサス評価**
3. **テクニカル分析**
4. **ファンダメンタルズ & バリュエーション評価**
5. **ニュース・外部要因の影響**
6. **投資戦略シナリオ**
"""
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

                        st.subheader("🤖 AI総合診断レポート（テクニカル × ファンダ × コンセンサス）")
                        st.write(response_text)

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

# ----------------------------------------------------
# TAB 2: TOPIX 100 買い推奨スクリーニング
# ----------------------------------------------------
with tab2:
    st.subheader("🏆 TOPIX 100（日本の主力100社）一括スクリーニング")
    st.caption("テクニカル反発シグナル（RSI・BB下限・MACD好転） × 割安バリュエーション（PER/PBR/配当） × アナリスト目標株価乖離率（アップサイド）を自動採点し、TOP5銘柄を抽出・解説します。")

    screen_btn = st.button("⚡ TOPIX100 一括スキャンを実行", type="primary", key="screen_btn")

    if screen_btn:
        if not api_key:
            st.error("Gemini APIキーが設定されていません。StreamlitのSecretsを確認してください。")
        else:
            progress_bar = st.progress(0, text="TOPIX100銘柄の株価データを一括取得中...")
            
            try:
                # 1. 一括ダウンロード
                df_all = yf.download(TOPIX100_SYMBOLS, period="150d", interval="1d", progress=False)
                df_close = df_all['Close']
                
                progress_bar.progress(30, text="テクニカル指標の判定と一次フィルタリング中...")
                
                # 2. テクニカル一次スクリーニング
                candidates = []
                for s in TOPIX100_SYMBOLS:
                    if s not in df_close.columns:
                        continue
                    close = df_close[s].dropna()
                    if len(close) < 75:
                        continue
                    latest_price = close.iloc[-1]
                    sma25 = close.rolling(25).mean().iloc[-1]
                    sma75 = close.rolling(75).mean().iloc[-1]
                    std25 = close.rolling(25).std().iloc[-1]
                    bb_lower = sma25 - (std25 * 2)
                    bb_upper = sma25 + (std25 * 2)

                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = (100 - (100 / (1 + rs))).iloc[-1]

                    exp12 = close.ewm(span=12, adjust=False).mean()
                    exp26 = close.ewm(span=26, adjust=False).mean()
                    macd = exp12 - exp26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    hist = macd - signal
                    latest_hist = hist.iloc[-1]
                    prev_hist = hist.iloc[-2]

                    tech_score = 0
                    if 30 <= rsi <= 45:
                        tech_score += 25
                    elif 45 < rsi <= 55:
                        tech_score += 15
                    if latest_price <= sma25 and latest_price >= bb_lower:
                        tech_score += 20
                    if latest_hist > prev_hist:
                        tech_score += 20
                    if close.iloc[-1] > sma75:
                        tech_score += 15

                    if tech_score >= 50:
                        candidates.append({
                            "symbol": s,
                            "code": s.replace(".T", ""),
                            "price": latest_price,
                            "rsi": rsi,
                            "tech_score": tech_score
                        })

                progress_bar.progress(60, text=f"候補 {len(candidates)} 銘柄のアナリスト目標株価・財務データを照合中...")

                # 3. ファンダメンタルズ & コンセンサス照合
                final_list = []
                for cand in candidates:
                    ticker = yf.Ticker(cand['symbol'])
                    info = ticker.info if hasattr(ticker, "info") else {}
                    target_price = info.get('targetMeanPrice', None)
                    num_analysts = info.get('numberOfAnalystOpinions', 0)
                    recommendation = info.get('recommendationKey', 'none').replace('_', ' ').title()

                    if not target_price or cand['price'] <= 0:
                        continue

                    upside_pct = ((target_price - cand['price']) / cand['price']) * 100
                    name = info.get('shortName', info.get('longName', cand['code']))
                    pe = info.get('trailingPE', info.get('forwardPE', None))

                    book_value = info.get('bookValue', None)
                    raw_pbr = info.get('priceToBook', None)
                    pbr = (cand['price'] / book_value) if (book_value and book_value > 0) else raw_pbr

                    div_rate = info.get('dividendRate', None)
                    div_yield = (div_rate / cand['price']) * 100 if div_rate else None

                    fund_score = 0
                    if upside_pct >= 20:
                        fund_score += 35
                    elif upside_pct >= 10:
                        fund_score += 20
                    elif upside_pct > 0:
                        fund_score += 10
                    if 'Buy' in recommendation:
                        fund_score += 15
                    if pe and pe <= 15:
                        fund_score += 15
                    if pbr and pbr <= 1.2:
                        fund_score += 15
                    if div_yield and div_yield >= 3.0:
                        fund_score += 10

                    total_score = cand['tech_score'] + fund_score
                    final_list.append({
                        "銘柄名": name,
                        "コード": cand['code'],
                        "現在値(円)": round(cand['price'], 1),
                        "アナリスト目標株価": round(target_price, 0),
                        "乖離率(%)": round(upside_pct, 1),
                        "コンセンサス": recommendation,
                        "カバー人数": num_analysts,
                        "PER(倍)": round(pe, 1) if pe else "N/A",
                        "PBR(倍)": round(pbr, 2) if pbr else "N/A",
                        "配当利回り(%)": round(div_yield, 2) if div_yield else "N/A",
                        "RSI(14日)": round(cand['rsi'], 1),
                        "総合スコア": total_score
                    })
                    time.sleep(0.05)

                progress_bar.progress(85, text="Geminiが上位5銘柄の投資戦略レポートを作成中...")

                # 4. TOP5抽出 ＆ Gemini分析
                df_result = pd.DataFrame(final_list)
                df_top5 = df_result.sort_values(by="総合スコア", ascending=False).head(5)

                st.success("✅ スクリーニング完了！")
                st.dataframe(df_top5.reset_index(drop=True), use_container_width=True)

                top5_summary = df_top5.to_string(index=False)
                prompt = f"""
あなたは百戦錬磨のシニア株式ストラテジストです。
日本の主力100社（TOPIX 100）を対象に、【テクニカル反発シグナル × 割安バリュエーション × アナリスト目標株価のアップサイド余地】で多角的スクリーニングを行い、抽出された以下の【買い推奨上位5銘柄】について、プロ投資家向けの選定理由・投資戦略レポートを作成してください。

【抽出された上位5銘柄データ】
{top5_summary}

---
以下の構成で詳細かつメリハリのある解説を行ってください：
1. **本日のスクリーニング総括**（全体地合いと、今回選定された5社に共通するテーマ性・セクター傾向）
2. **各銘柄の買い推奨理由 & エントリー戦略（5銘柄それぞれ個別に）**
   - **選定理由**: なぜ今が買い場なのか（テクニカルの反発サイン、PER/PBRや配当の魅力、アナリスト目標株価との乖離）
   - **実践シナリオ**: 想定エントリーポイント、ターゲット上値目標、下値サポート（損切り目安）
3. **ポートフォリオ構築のアドバイス & リスク管理**
"""
                client = genai.Client(api_key=api_key)
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

                progress_bar.progress(100, text="完了！")
                time.sleep(0.5)
                progress_bar.empty()

                st.subheader("📝 Gemini プロフェッショナル投資戦略レポート")
                st.write(response_text)

            except Exception as e:
                st.error(f"スクリーニング中にエラーが発生しました: {str(e)}")
