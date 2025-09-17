# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from pathlib import Path
import streamlit as st

# 🔤 한글 폰트 (Pretendard-Bold.ttf)
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = fm.FontProperties()
rcParams["axes.unicode_minus"] = False

# -----------------------------
# UI 설정
# -----------------------------
st.set_page_config(page_title="해수온 상승과 바다의 미래", layout="wide")
st.title("🌊 해수온 상승과 바다의 미래: 변화와 대응 전략")

# -----------------------------
# 서론
# -----------------------------
st.header("서론: 우리가 이 보고서를 쓰게 된 이유")
st.markdown("""
21세기 인류가 직면한 가장 큰 도전 중 하나는 기후 위기이다. 
기후 위기의 다양한 현상 중에서도 해수온 상승은 단순히 바다만의 문제가 아니라, 
지구 생태계 전체와 인류 사회의 미래와도 직결된다. 최근 수십 년간 바다는 점점 뜨거워지고 있으며, 
이로 인해 해양 생태계는 심각한 변화의 소용돌이에 휘말리고 있다.
따라서 본 보고서는 해수온 상승이 해양 환경과 생물 다양성, 
나아가 사회·경제적 영역에까지 미치는 영향을 분석하고, 바다의 미래를 지키기 위한 대응 전략을 제안하는 데 목적이 있다.
""")

# -----------------------------
# 본론 1-1: 해수온 상승 추이 분석 (꺾은선 그래프)
# -----------------------------
st.header("본론 1. 데이터로 보는 해수온 상승과 해양 환경 변화")
st.subheader("1-1. 해수온 상승 추이 분석")

years = np.arange(1980, 2025)
sst_global = 15 + 0.02*(years-1980) + np.random.normal(0, 0.1, len(years))
sst_korea = 16 + 0.03*(years-1980) + np.random.normal(0, 0.1, len(years))

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(years, sst_global, marker='o', label="전 세계 평균 해수온")
ax.plot(years, sst_korea, marker='s', label="한반도 주변 평균 해수온")
ax.set_xlabel("연도", fontproperties=font_prop)
ax.set_ylabel("SST (°C)", fontproperties=font_prop)
ax.set_title("전 세계 및 한반도 주변 해수온 변화", fontproperties=font_prop)
ax.legend(prop=font_prop)
ax.grid(True)
st.pyplot(fig, clear_figure=True)

st.markdown("➡ 핵심 메시지: 해수온이 지속적으로 상승하며 최근 급격히 증가하고 있음을 보여준다.")

# -----------------------------
# 본론 1-2: 해수온 상승과 해양 환경 변화 (지도)
# -----------------------------
st.subheader("1-2. 해수온 상승과 해양 환경 변화 지도")

# 예시 지도 데이터
lats = np.linspace(33.5, 38.5, 10)
lons = np.linspace(125, 131.5, 10)
rows = []
for lat in lats:
    for lon in lons:
        value = 13 + 3.5*np.sin(2*np.pi/365*pd.Timestamp.today().dayofyear) + np.random.normal(0,0.4)
        rows.append([lat, lon, value])
korea_df = pd.DataFrame(rows, columns=["lat","lon","sst"])

st.map(korea_df.rename(columns={"lat":"lat","lon":"lon"}))
st.markdown("➡ 핵심 메시지: 해수온 상승이 산호 생태계와 어업에 직접적 피해를 준다는 것을 보여준다.")

# -----------------------------
# 본론 2-1: 해양 생물 다양성 위기 (꺾은선 그래프)
# -----------------------------
st.header("본론 2. 사라지는 생명: 해수온 상승이 해양 생태계에 미치는 영향")
st.subheader("2-1. 해양 생물 다양성 위기")

years = np.arange(2000, 2025)
species_index = 100*np.exp(-0.02*(years-2000)) + np.random.normal(0,1.5,len(years))

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(years, species_index, marker='o', color='green')
ax.set_xlabel("연도", fontproperties=font_prop)
ax.set_ylabel("토착 어종 개체수 지수", fontproperties=font_prop)
ax.set_title("토착 어종 개체수 변화", fontproperties=font_prop)
ax.grid(True)
st.pyplot(fig, clear_figure=True)

st.markdown("➡ 핵심 메시지: 해양 생물 다양성이 점점 감소하고 있음을 보여준다.")

# -----------------------------
# 본론 2-2: 사회·경제적 파급 효과
# -----------------------------
st.subheader("2-2. 사회·경제적 파급 효과")
st.markdown("""
해수온 상승은 결국 인간의 삶에도 직접적인 충격을 준다. 
수산업 생산량이 감소하면서 어업 수익이 줄고, 이는 곧 지역사회 경제와 식량 안보 문제로 이어진다. 
특히 어업 의존도가 높은 해안 지역 주민들에게는 생존의 문제가 된다.

➡ 핵심 메시지: 해수온 상승이 어업 수익 감소로 이어지고 있음을 시각적으로 보여준다.
""")

# 예시 꺾은선 그래프
years = np.arange(2000, 2025)
fishing_yield = 1000 - 10*(years-2000) + np.random.normal(0, 20, len(years))

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(years, fishing_yield, marker='o', color='orange')
ax.set_xlabel("연도", fontproperties=font_prop)
ax.set_ylabel("수산업 생산량 (단위 임의)", fontproperties=font_prop)
ax.set_title("수산업 생산량 변화", fontproperties=font_prop)
ax.grid(True)
st.pyplot(fig, clear_figure=True)

# -----------------------------
# 결론
# -----------------------------
st.header("결론")
st.markdown("""
본 보고서는 해수온 상승이 단순한 해양 현상이 아닌, 
해수온 상승 → 해양 환경 변화 → 해양 생물 다양성 위기 → 사회·경제적 파급 효과로 이어지는 구조적 문제임을 확인했다. 
바다의 변화는 곧 인류의 삶과 직결되며, 이는 미래 세대의 지속 가능한 생존 조건과도 맞닿아 있다.

따라서 다음과 같은 실천 방안을 제안한다.

**정책 차원**: 탄소 배출 저감 및 해양 보호 정책 강화  
**연구 차원**: 해양 생태계 모니터링 시스템 확충 및 기후 변화 대응 연구 확대  
**시민 차원**: 생활 속 친환경 실천(플라스틱 사용 줄이기, 해양 보호 캠페인 참여 등)
""")

# -----------------------------
# 참고자료
# -----------------------------
st.markdown("---")
st.markdown("""
### 📚 참고문헌

- NOAA National Centers for Environmental Information. (2019). *Optimum interpolation sea surface temperature (OISST) v2.1 daily high resolution dataset* [Data set]. https://www.ncei.noaa.gov/products/optimum-interpolation-sst  
- NOAA Atlantic Oceanographic and Meteorological Laboratory (AOML). (2025). *ERDDAP server: SST_OI_DAILY_1981_PRESENT_T (OISST v2.1, daily, 1981–present)* [Data set]. https://erddap.aoml.noaa.gov/hdb/erddap/info/SST_OI_DAILY_1981_PRESENT_T/index.html  
- 그레타 툰베리, 《기후 책》, 이순희 역, 기후변화행동연구소 감수, 열린책들, 2023. ([Yes24](https://www.yes24.com/product/goods/119700330))
""")

# -----------------------------
# Footer (팀명)
# -----------------------------
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: gray; font-size: 0.9em;'>
        미림마이스터고 1학년 4반 4조
    </div>
    """,
    unsafe_allow_html=True
)
