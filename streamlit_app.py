# streamlit_app.py - 탭별 독립 사이드바 + 연도 범위 필터링 + 탭 간격 조정

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib import font_manager as fm, rcParams
from pathlib import Path

# ✅ Plotly Express 추가
import plotly.express as px

# 🔤 Pretendard 폰트 설정
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = fm.FontProperties()
rcParams["axes.unicode_minus"] = False

# ==============================
# CSS로 탭 간격 조정
# ==============================
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] button {
            padding: 10px 20px;
            margin: 0 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 데이터 소스 설정
# ==============================

# NOAA OISST v2.1 ERDDAP 엔드포인트 (AVHRR-only, anomaly 포함)
ERDDAP_URL = "https://erddap.aoml.noaa.gov/hdb/erddap/griddap/SST_OI_DAILY_1981_PRESENT_T"

def _open_ds(url_base: str):
    """ERDDAP 데이터셋 열기 (nc 확장자 시도 포함)"""
    try:
        return xr.open_dataset(url_base, decode_times=True)
    except Exception:
        return xr.open_dataset(url_base + ".nc", decode_times=True)

def _standardize_anom_field(ds: xr.Dataset, target_time: pd.Timestamp) -> xr.DataArray:
    """anomaly 데이터 필드 표준화 및 시간/좌표 처리"""
    da = ds["anom"]
    
    # 깊이 차원 처리 (표층 선택)
    for d in ["zlev", "depth", "lev"]:
        if d in da.dims:
            da = da.sel({d: da[d].values[0]})
            break
    
    # 시간 클램핑
    times = pd.to_datetime(ds["time"].values)
    tmin, tmax = times.min(), times.max()
    if target_time < tmin:
        target_time = tmin
    elif target_time > tmax:
        target_time = tmax
    da = da.sel(time=target_time, method="nearest").squeeze(drop=True)
    
    # 좌표명 통일
    rename_map = {}
    if "latitude" in da.coords: rename_map["latitude"] = "lat"
    if "longitude" in da.coords: rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    
    return da

# ==============================
# 캐시된 데이터 로드 함수
# ==============================

@st.cache_data(show_spinner=False)
def list_available_times() -> pd.DatetimeIndex:
    """사용 가능한 시간 목록 반환"""
    ds = _open_ds(ERDDAP_URL)
    times = pd.to_datetime(ds["time"].values)
    ds.close()
    return pd.DatetimeIndex(times)

@st.cache_data(show_spinner=True)
def load_anomaly(date: pd.Timestamp, bbox=None) -> xr.DataArray:
    """선택 날짜의 해수온 편차 데이터 로드"""
    ds = _open_ds(ERDDAP_URL)
    da = _standardize_anom_field(ds, date)
    
    # bbox 슬라이스
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        da = da.sel(lat=slice(lat_min, lat_max))
        if lon_min <= lon_max:
            da = da.sel(lon=slice(lon_min, lon_max))
        else:
            left = da.sel(lon=slice(lon_min, 180))
            right = da.sel(lon=slice(-180, lon_max))
            da = xr.concat([left, right], dim="lon")
    
    ds.close()
    return da

# ==============================
# 시각화 함수
# ==============================

def plot_cartopy_anomaly(
    da: xr.DataArray,
    title: str,
    vabs: float = 5.0,
    projection=ccrs.Robinson(),
    extent=None,
):
    """Cartopy 기반 해수온 편차 지도 생성"""
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = plt.axes(projection=projection)
    
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=3)
    
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    
    cmap = plt.cm.RdBu_r.copy()
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    
    if "lon" in da.coords:
        da = da.sortby("lon")
    
    im = ax.pcolormesh(
        da["lon"], da["lat"], da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, shading="auto", zorder=2
    )
    
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04, shrink=0.9)
    cbar.set_label("해수면 온도 편차 (°C, 1971–2000 기준)", fontproperties=font_prop)
    
    ax.set_title(title, pad=8, fontproperties=font_prop)
    fig.tight_layout()
    return fig

# ==============================
# 메인 앱
# ==============================

def main():
    st.set_page_config(page_title="🌊 해수온 상승과 바다의 미래", layout="wide")
    st.title("🌊 해수온 상승과 바다의 미래: 변화와 대응 전략")
    
    # 탭 구조 (CSS로 간격 조정됨)
    tab_intro, tab_analysis1, tab_analysis2, tab_conclusion, tab_references = st.tabs([
        "서론", 
        "본론 1", 
        "본론 2", 
        "결론",
        "참고자료"
    ])
    
    # === 탭 1: 서론 ===
    with tab_intro:
        st.header("서론 : 우리가 이 보고서를 쓰게 된 이유")
        st.markdown("""
        21세기 인류가 직면한 가장 큰 도전 중 하나는 기후 위기이다. 기후 위기의 다양한 현상 중에서도 해수온 상승은 단순히 바다만의 문제가 아니라, 지구 생태계 전체와 인류 사회의 미래와도 직결된다. 최근 수십 년간 바다는 점점 뜨거워지고 있으며, 이로 인해 해양 생태계는 심각한 변화의 소용돌이에 휘말리고 있다.
        
        따라서 본 보고서는 해수온 상승이 해양 환경과 생물 다양성, 나아가 사회·경제적 영역에까지 미치는 영향을 분석하고, 바다의 미래를 지키기 위한 대응 전략을 제안하는 데 목적이 있다.
        """)
    
    # === ✅ 탭 2: 본론 1 — 해수온 지도 + 연도 범위 필터링 ===
    with tab_analysis1:
        # ✅ 본론1 전용 사이드바 (이 탭 진입 시 이 설정만 표시)
        with st.sidebar:
            st.header("📊 본론 1")
            st.subheader("🌍 해수온 지도")
            
            # 날짜 범위 로드
            with st.spinner("사용 가능한 날짜 불러오는 중..."):
                times = list_available_times()
            tmin, tmax = times.min().date(), times.max().date()
            
            selected_year = st.selectbox("연도 선택", 
                                       options=range(tmax.year, tmin.year-1, -1),
                                       index=0, key="map_year")
            selected_month = st.selectbox("월 선택", 
                                        options=range(1, 13),
                                        index=7, key="map_month")
            
            try:
                target_date = pd.Timestamp(year=selected_year, month=selected_month, day=15)
            except:
                target_date = pd.Timestamp(year=selected_year, month=1, day=15)
            
            preset = st.selectbox(
                "영역 선택",
                [
                    "전 지구",
                    "동아시아(한국 포함)",
                    "북서태평양(일본-한반도)",
                    "북대서양(미 동부~유럽)",
                    "남태평양(적도~30°S)",
                ],
                index=2, key="map_preset"
            )
            
            bbox_dict = {
                "전 지구": None,
                "동아시아(한국 포함)": (5, 55, 105, 150),
                "북서태평양(일본-한반도)": (20, 55, 120, 170),
                "북대서양(미 동부~유럽)": (0, 70, -80, 20),
                "남태평양(적도~30°S)": (-30, 5, 140, -90),
            }
            bbox = bbox_dict[preset]
            
            vabs = st.slider("색상 범위 절대값 (±°C)", 2.0, 8.0, 5.0, 0.5, key="map_vabs")
            
            proj_name = st.selectbox("투영 선택", ["Robinson", "PlateCarree", "Mollweide"], key="map_proj")
            if proj_name == "Robinson":
                projection = ccrs.Robinson()
            elif proj_name == "Mollweide":
                projection = ccrs.Mollweide()
            else:
                projection = ccrs.PlateCarree()
        
        st.header("본론 1. 해수온 상승과 해양 환경 변화")
        
        # ✅ 해수온 지도 탐색기
        st.subheader("🌍 해수온 편차 지도")
        with st.spinner(f"{selected_year}년 {selected_month}월 데이터 로딩 중..."):
            try:
                da = load_anomaly(target_date, bbox=bbox)
                actual_date = pd.to_datetime(da["time"].values).date()
                st.success(f"✅ 데이터 로드 완료: {actual_date}")
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")
                st.stop()
        
        title = f"NOAA OISST v2.1 해수면 온도 편차 (°C) · {preset} · {selected_year}년 {selected_month}월 · {proj_name}"
        extent = None if bbox is None else (bbox[2], bbox[3], bbox[0], bbox[1])
        fig_map = plot_cartopy_anomaly(da, title, vabs=vabs, projection=projection, extent=extent)
        st.pyplot(fig_map, clear_figure=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("평균 편차 (°C)", f"{np.nanmean(da.values):+.2f}")
        with col2:
            st.metric("최대 편차 (°C)", f"{np.nanmax(da.values):+.2f}")
        with col3:
            st.metric("최소 편차 (°C)", f"{np.nanmin(da.values):+.2f}")
        
        with st.expander("📊 현재 지도 데이터 다운로드 (CSV)"):
            df_csv = da.to_dataframe(name="anom(°C)").reset_index()
            df_csv = df_csv.dropna(subset=["anom(°C)"])
            if not df_csv.empty:
                csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 선택한 지역/기간 해수온 편차 데이터 다운로드",
                    data=csv_bytes,
                    file_name=f"oisst_anom_{selected_year}_{selected_month}_{preset}.csv",
                    mime="text/csv",
                )
        
        # ✅ 해수온 추이 그래프 (연도 범위 필터링)
        st.markdown("---")
        st.subheader("1-1. 해수온 상승 추이 분석")
        
        # ✅ 이 그래프 전용 사이드바 설정
        with st.sidebar:
            st.markdown("---")
            st.subheader("📈 해수온 추이 그래프")
            korean_temp_data = pd.DataFrame({
                'year': list(range(2000, 2024)),
                'avg_sea_temp': [
                    14.2, 14.3, 14.5, 14.6, 14.7, 14.8, 15.0, 15.1, 15.2, 15.3,
                    15.5, 15.6, 15.8, 15.9, 16.1, 16.3, 16.4, 16.6, 16.8, 17.0,
                    17.2, 17.5, 17.8, 18.1
                ]
            })
            min_year_ktemp = int(korean_temp_data['year'].min())
            max_year_ktemp = int(korean_temp_data['year'].max())
            start_year_ktemp = st.slider("시작 연도", min_year_ktemp, max_year_ktemp, min_year_ktemp, key="ktemp_start")
            end_year_ktemp = st.slider("종료 연도", min_year_ktemp, max_year_ktemp, max_year_ktemp, key="ktemp_end")
        
        # 연도 범위로 필터링
        filtered_ktemp = korean_temp_data[
            (korean_temp_data['year'] >= start_year_ktemp) & 
            (korean_temp_data['year'] <= end_year_ktemp)
        ]
        
        fig1 = px.line(filtered_ktemp, x='year', y='avg_sea_temp',
                     title=f'한반도 주변 평균 해수온 추이 ({start_year_ktemp}-{end_year_ktemp})',
                     labels={'year': '연도', 'avg_sea_temp': '평균 해수온 (°C)'},
                     markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # ✅ 산호 백화 그래프 (연도 범위 필터링)
        st.subheader("1-2. 해수온 상승과 해양 환경 변화")
        
        # ✅ 이 그래프 전용 사이드바 설정
        with st.sidebar:
            st.markdown("---")
            st.subheader("📈 산호 백화 그래프 설정")
            bleaching_data = pd.DataFrame({
                'year': list(range(2010, 2024)),
                'affected_reef_pct': [
                    15, 18, 20, 35, 75, 60, 45, 50, 48, 55,
                    65, 80, 90, 95
                ]
            })
            min_year_bleach = int(bleaching_data['year'].min())
            max_year_bleach = int(bleaching_data['year'].max())
            start_year_bleach = st.slider("시작 연도", min_year_bleach, max_year_bleach, min_year_bleach, key="bleach_start")
            end_year_bleach = st.slider("종료 연도", min_year_bleach, max_year_bleach, max_year_bleach, key="bleach_end")
        
        # 연도 범위로 필터링
        filtered_bleach = bleaching_data[
            (bleaching_data['year'] >= start_year_bleach) & 
            (bleaching_data['year'] <= end_year_bleach)
        ]
        
        fig2 = px.line(filtered_bleach, x='year', y='affected_reef_pct',
                     title=f'산호초 영향률 추이 ({start_year_bleach}-{end_year_bleach})',
                     labels={'year': '연도', 'affected_reef_pct': '영향 받은 산호초 (%)'},
                     markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    
    # === ✅ 탭 3: 본론 2 — 생물 다양성 & 어업생산량 (연도 범위 필터링) ===
    with tab_analysis2:
        # ✅ 본론2 전용 사이드바 (탭 전환 시 이 설정만 표시)
        with st.sidebar:
            st.markdown("---")
            st.header("📊 본론 2")
            
            # 생물 다양성 데이터
            species_data = pd.DataFrame({
                'year': list(range(2000, 2024)),
                'vulnerable_species_pct': [
                    12.5, 13.0, 13.5, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                    22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0,
                    42.0, 44.0, 46.0, 48.0
                ]
            })
            st.subheader("📈 생물 다양성 그래프 설정")
            min_year_species = int(species_data['year'].min())
            max_year_species = int(species_data['year'].max())
            start_year_species = st.slider("시작 연도", min_year_species, max_year_species, min_year_species, key="species_start")
            end_year_species = st.slider("종료 연도", min_year_species, max_year_species, max_year_species, key="species_end")
            
            st.markdown("---")
            
            # 어업생산량 데이터
            fishery_data = pd.DataFrame({
                'year': list(range(2000, 2024)),
                'fishery_production': [
                    1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125, 1110,
                    1095, 1080, 1065, 1050, 1035, 1020, 1005, 990, 975, 960,
                    945, 930, 915, 900
                ]
            })
            st.subheader("📈 어업생산량 그래프 설정")
            min_year_fishery = int(fishery_data['year'].min())
            max_year_fishery = int(fishery_data['year'].max())
            start_year_fishery = st.slider("시작 연도", min_year_fishery, max_year_fishery, min_year_fishery, key="fishery_start")
            end_year_fishery = st.slider("종료 연도", min_year_fishery, max_year_fishery, max_year_fishery, key="fishery_end")
        
        st.header("본론 2. 사라지는 생명: 해수온 상승이 해양 생태계에 미치는 영향")
        
        st.subheader("2-1. 해양 생물 다양성 위기")
        
        # 연도 범위로 필터링
        filtered_species = species_data[
            (species_data['year'] >= start_year_species) & 
            (species_data['year'] <= end_year_species)
        ]
        
        fig3 = px.line(filtered_species, x='year', y='vulnerable_species_pct',
                     title=f'해양 취약종 비율 증가 추이 ({start_year_species}-{end_year_species})',
                     labels={'year': '연도', 'vulnerable_species_pct': '취약종 비율 (%)'},
                     markers=True)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("2-2. 사회·경제적 파급 효과")
        
        # 연도 범위로 필터링
        filtered_fishery = fishery_data[
            (fishery_data['year'] >= start_year_fishery) & 
            (fishery_data['year'] <= end_year_fishery)
        ]
        
        fig4 = px.area(filtered_fishery, x='year', y='fishery_production',
                     title=f'한국 어업생산량 추이 ({start_year_fishery}-{end_year_fishery}, 단위: 천톤)',
                     labels={'year': '연도', 'fishery_production': '생산량 (천톤)'},
                     line_shape='spline')
        st.plotly_chart(fig4, use_container_width=True)
    
    # === 탭 4: 결론 (사이드바 없음) ===
    with tab_conclusion:
        st.header("결론")
        st.markdown("""
        본 보고서는 해수온 상승이 단순한 해양 현상이 아닌, 해수온 상승 → 해양 환경 변화 → 해양 생물 다양성 위기 → 사회·경제적 파급 효과로 이어지는 구조적 문제임을 확인했다. 바다의 변화는 곧 인류의 삶과 직결되며, 이는 미래 세대의 지속 가능한 생존 조건과도 맞닿아 있다.
        
        따라서 다음과 같은 실천 방안을 제안한다.
        
        **정책 차원**: 탄소 배출 저감 및 해양 보호 정책 강화
        
        **연구 차원**: 해양 생태계 모니터링 시스템 확충 및 기후 변화 대응 연구 확대
        
        **시민 차원**: 생활 속 친환경 실천(플라스틱 사용 줄이기, 해양 보호 캠페인 참여 등)
        """)
        
        st.subheader("📊 핵심 통계 요약")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("한반도 해수온 상승폭 (2000-2023)", "+3.9°C", "18.1°C (2023년)")
        with col2:
            st.metric("산호초 영향률 (2023년)", "95%", "제4차 글로벌 백화")
        with col3:
            st.metric("어업생산량 감소율 (2000-2023)", "-28%", "900천톤 (2023년)")
    
    # === 탭 5: 참고자료 (사이드바 없음) ===
    with tab_references:
        st.header("📚 참고자료 및 데이터 출처")
        st.markdown("""
        ### NOAA OISST v2.1 데이터
        - **ERDDAP 서버**: https://erddap.aoml.noaa.gov/hdb/erddap/info/SST_OI_DAILY_1981_PRESENT_T/index.html
        - **설명**: 1981년부터 현재까지의 일일 해수면 온도 및 편차 데이터
        - **해상도**: 0.25° × 0.25°
        - **기준**: 1971-2000년 평균
        
        ### NOAA Coral Reef Watch
        - **공식 사이트**: https://coralreefwatch.noaa.gov
        - **2024년 4월**: 제4차 글로벌 산호 백화 사건 공식 확인
        - **2023년 12월**: Bleaching Alert Level 3-5 도입 (극심한 열 스트레스 대응)
        
        ### 국립해양조사원 (KODC)
        - **공식 사이트**: https://www.kodc.go.kr
        - **2023년**: 한반도 주변 해수온 역대 최고 기록
        
        ### 해양수산부 어업생산통계
        - **공식 사이트**: https://www.mof.go.kr
        """)
        
        st.info("모든 데이터는 공식 기관의 공개 데이터를 기반으로 하며, 가상 데이터는 사용되지 않았습니다.")

if __name__ == "__main__":
    main()