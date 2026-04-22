"""
Yağmur Hasadı Simülasyon Platformu
Ana Streamlit Uygulaması (Map drag-drop + Gerçek Meteoroloji)

Çalıştırın: streamlit run app.py
"""

import json
import math
import time
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import folium
from streamlit_folium import st_folium

import config
from modules.simulation_engine import SimulationEngine, replay_tank_for_rainfall
from modules.visualization import (
    Scene3D,
    TimeSeriesGraphs,
    build_pydeck_overlay,
)


# =========================================================================
# Sayfa yapılandırması
# =========================================================================
st.set_page_config(
    page_title="Yağmur Hasadı Simülasyon Platformu",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# Oturum durumu varsayılanları
# =========================================================================
_DEFAULT_LOCATION = (39.9334, 32.8597)  # Ankara

_SESSION_DEFAULTS = {
    "selected_location": _DEFAULT_LOCATION,
    "sim_results": None,
    "simulation_engine": None,
    "simulation_run": False,
    "weather_data": None,
    "weather_source": None,
    "frame_idx": 0,
    "is_playing": False,
    "last_map_ts": 0.0,
    "picker_key_counter": 0,
    "lat_input": _DEFAULT_LOCATION[0],
    "lon_input": _DEFAULT_LOCATION[1],
    "anim_scope": "Aylık",
    "anim_month": 1,
    "_last_scope": "Aylık",
    "_last_month": 1,
}

for _key, _val in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# =========================================================================
# Yardımcılar
# =========================================================================
@st.cache_data(show_spinner=False)
def fetch_weather_data(
    latitude: float, longitude: float, year: int = 2024
) -> Optional[pd.DataFrame]:
    """Open-Meteo Archive API'dan günlük yağış ve sıcaklık verisi çeker."""
    try:
        import requests

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean",
            "timezone": "auto",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "daily" not in data:
            return None

        df = pd.DataFrame(
            {
                "prcp": data["daily"]["precipitation_sum"],
                "tmax": data["daily"]["temperature_2m_max"],
                "tmin": data["daily"]["temperature_2m_min"],
                "tavg": data["daily"]["temperature_2m_mean"],
            },
            index=pd.to_datetime(data["daily"]["time"]),
        )
        return df.fillna(0)
    except Exception:
        return None


def draw_worker_simulation(worker_count: int, water_available: float, water_needed: float):
    """İşçi dağılımı ve su dengesi için basit matplotlib görseli."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect("equal")
    ax1.set_title(f"👥 İşçi Dağılımı ({worker_count} kişi)", fontsize=14, fontweight="bold")
    ax1.axis("off")

    rng = np.random.RandomState(42)
    for _ in range(min(worker_count, 100)):
        x = rng.uniform(1, 9)
        y = rng.uniform(1, 9)
        ax1.add_patch(patches.Circle((x, y), 0.2, color="#FF6B6B", alpha=0.7))
    if worker_count > 100:
        ax1.text(5, 0.5, f"+ {worker_count - 100} daha", ha="center", fontsize=12, fontweight="bold")

    ax2.barh(
        ["Mevcut Su", "İhtiyaç Duyulan"],
        [water_available, water_needed],
        color=["#51CF66", "#FF8C42"],
    )
    ax2.set_xlabel("Su Miktarı (Litre)", fontsize=12)
    ax2.set_title("💧 Günlük Su Dengesi", fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate([water_available, water_needed]):
        ax2.text(v + 100, i, f"{v:,.0f}L", va="center", fontweight="bold")
    balance = water_available - water_needed
    balance_color = "#51CF66" if balance >= 0 else "#FF6B6B"
    ax2.text(
        0.5, -0.15, f"Denge: {balance:+,.0f}L",
        transform=ax2.transAxes, ha="center",
        fontsize=12, fontweight="bold", color=balance_color,
    )
    plt.tight_layout()
    return fig


def _extract_marker_location(st_data: Optional[dict]) -> Optional[Tuple[float, float]]:
    """st_folium farklı sürümlerde koordinatı farklı anahtarda döner; hepsini dene."""
    if not st_data:
        return None
    for key in ("last_object_clicked", "last_object_clicked_popup", "last_clicked"):
        entry = st_data.get(key)
        if isinstance(entry, dict):
            lat = entry.get("lat") or entry.get("latitude")
            lon = entry.get("lng") or entry.get("longitude") or entry.get("lon")
            if lat is not None and lon is not None:
                try:
                    return float(lat), float(lon)
                except (TypeError, ValueError):
                    continue
    # Marker drag değişimi bazı sürümlerde "all_drawings" altında gelir.
    drawings = st_data.get("all_drawings")
    if isinstance(drawings, list) and drawings:
        geom = drawings[-1].get("geometry") if isinstance(drawings[-1], dict) else None
        if isinstance(geom, dict) and geom.get("type") == "Point":
            coords = geom.get("coordinates") or []
            if len(coords) >= 2:
                return float(coords[1]), float(coords[0])
    return None


def _status_badge(label: str, value: str, ok: bool) -> str:
    color = "#16A34A" if ok else "#9CA3AF"
    bg = "#DCFCE7" if ok else "#F3F4F6"
    icon = "✓" if ok else "…"
    return (
        f"<div style='padding:10px 14px;border-radius:10px;background:{bg};"
        f"border:1px solid {color};color:#111827;'>"
        f"<div style='font-size:0.75rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;'>{label}</div>"
        f"<div style='font-size:0.95rem;font-weight:600;color:{color};'>{icon} {value}</div>"
        f"</div>"
    )


# =========================================================================
# Başlık
# =========================================================================
st.title("🌧️ Yağmur Hasadı Simülatörü")
st.caption(
    "Haritaya binayı sürükleyip bırakın, konumun gerçek yağış verisiyle 365 günlük "
    "simülasyonu çalıştırın ve haritanın üstünde canlı depo/yağmur animasyonunu izleyin."
)


# =========================================================================
# Sidebar — parametreler
# =========================================================================
st.sidebar.header("⚙️ Parametreler")

çatı_alanı = st.sidebar.slider(
    "Çatı Alanı (m²)",
    min_value=100, max_value=200000,
    value=config.ROOF_AREA_DEFAULT, step=50,
    help="Su toplama için toplam çatı alanı",
)
çatı_verimlilik = st.sidebar.slider(
    "Toplama Verimliliği",
    min_value=0.0, max_value=1.0,
    value=config.ROOF_EFFICIENCY, step=0.05,
    help="Sistem verimliliği (0 = toplama yok, 1 = mükemmel)",
)
depo_kapasitesi = st.sidebar.slider(
    "Depo Kapasitesi (Litre)",
    min_value=5000, max_value=1000000,
    value=config.TANK_CAPACITY_DEFAULT, step=5000,
    help="Depolama tankının kapasitesi",
)
çalışan_sayısı = st.sidebar.slider(
    "Çalışan Sayısı",
    min_value=1, max_value=30000,
    value=config.WORKER_COUNT_DEFAULT, step=5,
    help="Su tüketen kişi sayısı",
)
tüketim_oranı = st.sidebar.slider(
    "Tüketim (L/çalışan/saat)",
    min_value=0.5, max_value=25.0,
    value=config.CONSUMPTION_PER_WORKER_PER_HOUR, step=0.5,
)
yağış_tohumu = st.sidebar.number_input(
    "Stokastik Yağış Tohumu",
    min_value=0, max_value=100_000, value=config.RAIN_SEED, step=1,
    help="Open-Meteo erişilemezse sentetik yağış için kullanılır.",
)

st.sidebar.markdown("### 💰 Ekonomi")
su_fiyatı = st.sidebar.number_input(
    "Su Fiyatı (₺/m³)",
    min_value=0.0, max_value=500.0, value=config.WATER_PRICE * 1000, step=1.0,
)
depo_maliyeti = st.sidebar.number_input(
    "Depo Maliyeti (₺)",
    min_value=1000, max_value=50_000_000, value=config.TANK_COST, step=500,
    help="Yağmur suyu deposunun tek seferlik satın alma bedeli.",
)
kurulum_maliyeti = st.sidebar.number_input(
    "Kurulum Maliyeti (₺)",
    min_value=500, max_value=10_000_000, value=config.INSTALLATION_COST, step=500,
    help="Boru, pompa, filtre, first-flush ayırıcı ve işçilik bedelinin toplamı.",
)
bakım_maliyeti = st.sidebar.number_input(
    "Yıllık Bakım Maliyeti (₺)",
    min_value=100, max_value=500_000, value=config.MAINTENANCE_COST, step=100,
)
with st.sidebar.expander("🧮 Geri Ödeme Varsayımları", expanded=False):
    su_fiyat_artışı = st.number_input(
        "Su Fiyatı Artışı (reel, %/yıl)",
        min_value=0.0, max_value=50.0,
        value=config.WATER_PRICE_ESCALATION * 100, step=0.5,
        help="Su tarifesinin genel enflasyon ÜSTÜNDEKİ yıllık reel artışı.",
    ) / 100.0
    iskonto_oranı = st.number_input(
        "İskonto Oranı (reel, %/yıl)",
        min_value=0.0, max_value=25.0,
        value=config.DISCOUNT_RATE * 100, step=0.5,
        help="Yatırımcının parasal zaman değeri — reel ağırlıklı sermaye maliyeti.",
    ) / 100.0
    sistem_ömrü = st.number_input(
        "Sistem Ömrü (yıl)",
        min_value=5, max_value=40,
        value=config.SYSTEM_LIFESPAN_YEARS, step=1,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "1. Haritaya binayı bırakın\n"
    "2. Parametreleri ayarlayın\n"
    "3. **Simülasyonu Başlat**"
)


# =========================================================================
# Durum çubuğu (üst)
# =========================================================================
loc = st.session_state.selected_location
loc_ok = loc is not None
loc_text = f"{loc[0]:.3f}°, {loc[1]:.3f}°" if loc_ok else "seçilmedi"

wx = st.session_state.weather_data
wx_ok = wx is not None and len(wx) > 0
wx_text = (
    f"{st.session_state.weather_source or 'Open-Meteo'} · {len(wx)} gün" if wx_ok else "bekleniyor"
)

sim_ok = bool(st.session_state.simulation_run and st.session_state.sim_results)
if sim_ok:
    _n_days_run = len(
        st.session_state.sim_results.get("daily_history", {}).get("day", []) or []
    )
    sim_text = f"{_n_days_run} gün çalıştı" if _n_days_run else "tamamlandı"
else:
    sim_text = "bekleniyor"

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(_status_badge("Konum", loc_text, loc_ok), unsafe_allow_html=True)
with s2:
    st.markdown(_status_badge("Meteoroloji", wx_text, wx_ok), unsafe_allow_html=True)
with s3:
    st.markdown(_status_badge("Simülasyon", sim_text, sim_ok), unsafe_allow_html=True)

st.markdown("")


# =========================================================================
# Harita + Overlay (iki kolonlu yerleşim)
# =========================================================================
left_col, right_col = st.columns([1, 1], gap="large")

# --------- Sol: Folium picker (draggable bina DivIcon) ---------
with left_col:
    st.subheader("1) Binayı Haritaya Sürükleyip Bırakın")

    lat_current, lon_current = st.session_state.selected_location

    # Harita marker'ı sürüklenince selected_location güncellenip rerun tetiklenir.
    # Streamlit, widget instantiate edildikten sonra aynı key'li session_state'i
    # yazmaya izin vermediği için senkronizasyonu number_input render edilmeden
    # ÖNCE burada yaparız (widget henüz oluşmadı → serbest yazabiliriz).
    if abs(float(st.session_state.lat_input) - lat_current) > 1e-6:
        st.session_state.lat_input = float(lat_current)
    if abs(float(st.session_state.lon_input) - lon_current) > 1e-6:
        st.session_state.lon_input = float(lon_current)

    # Lat/lon ince ayar (fallback — her zaman görünür). Kullanıcı input'u
    # değiştirince selected_location güncellenir ve harita yeniden oluşturulur.
    def _on_coord_change() -> None:
        new_loc = (
            float(st.session_state.lat_input),
            float(st.session_state.lon_input),
        )
        if new_loc != st.session_state.selected_location:
            st.session_state.selected_location = new_loc
            st.session_state.picker_key_counter += 1

    coord_c1, coord_c2 = st.columns(2)
    with coord_c1:
        st.number_input(
            "Enlem", format="%.4f", step=0.0010,
            key="lat_input", on_change=_on_coord_change,
        )
    with coord_c2:
        st.number_input(
            "Boylam", format="%.4f", step=0.0010,
            key="lon_input", on_change=_on_coord_change,
        )

    # Folium haritası + emoji DivIcon (parametrelere göre ölçeklenen bina ikonu).
    # Harita hem idle hem oynatma sırasında görünür kalır; oynatma sırasında
    # drag callback'ine ihtiyaç olmadığı için st_folium returned_objects=[]
    # ile çağrılır ve widget yerinde güncellenir (unmount/remount yok).
    center_lat, center_lon = st.session_state.selected_location
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    size_px = int(max(36, min(72, math.sqrt(max(10.0, float(çatı_alanı))) * 1.2)))
    icon_w = size_px + 20
    icon_h = size_px + 10
    icon_html = (
        "<div style=\"display:flex;flex-direction:column;align-items:center;"
        "pointer-events:auto;cursor:grab;user-select:none;\">"
        f"<div style=\"font-size:{size_px}px;line-height:1;"
        "text-shadow:0 2px 4px rgba(0,0,0,.45);\">🏢</div>"
        "</div>"
    )
    div_icon = folium.DivIcon(
        icon_size=(icon_w, icon_h),
        icon_anchor=(icon_w // 2, icon_h - 6),
        html=icon_html,
    )
    folium.Marker(
        [center_lat, center_lon],
        draggable=True,
        icon=div_icon,
        tooltip="Binayı sürükleyip bırakın",
    ).add_to(fmap)

    # st_folium her iki durumda da kullanılır; key stabil tutularak browser
    # tarafında widget yerinde güncellenir (unmount/remount yok → harita
    # oynatma sırasında da görünür kalır). Oynatma sırasında drag callback'i
    # gerekmez; returned_objects=[] ile rerun maliyeti en aza iner.
    picker_key = f"picker-{st.session_state.picker_key_counter}"
    if st.session_state.is_playing:
        st_folium(
            fmap,
            key=picker_key,
            width=None,
            height=430,
            returned_objects=[],
        )
        st.caption("Animasyon oynuyor; sürüklemek için Durdur'a basın.")
    else:
        st_data = st_folium(
            fmap,
            key=picker_key,
            width=None,
            height=430,
            returned_objects=[
                "last_object_clicked",
                "last_clicked",
                "center",
                "all_drawings",
            ],
        )

        marker_loc = _extract_marker_location(st_data)
        if marker_loc is not None:
            cur = st.session_state.selected_location
            if (abs(marker_loc[0] - cur[0]) > 1e-5) or (abs(marker_loc[1] - cur[1]) > 1e-5):
                # Sadece selected_location'i güncelle + rerun; widget'lar bir
                # sonraki run'da üstteki senkronizasyon bloğunda lat/lon_input
                # session_state'ini güncelleyecek.
                st.session_state.selected_location = marker_loc
                st.session_state.picker_key_counter += 1
                st.rerun()

        st.caption(
            "İpucu: Marker'ın üzerine gelip basılı tutarak sürükleyin; bırakınca konum güncellenir. "
            "Alternatif olarak yukarıdan enlem/boylam girebilirsiniz."
        )

    run_col, reset_col = st.columns([2, 1])
    with run_col:
        start = st.button(
            "▶ Simülasyonu Başlat", type="primary", use_container_width=True
        )
    with reset_col:
        if st.button("🔄 Sıfırla", use_container_width=True):
            st.session_state.sim_results = None
            st.session_state.simulation_engine = None
            st.session_state.simulation_run = False
            st.session_state.weather_data = None
            st.session_state.weather_source = None
            st.session_state.frame_idx = 0
            st.session_state.is_playing = False
            st.session_state.selected_location = _DEFAULT_LOCATION
            st.session_state.picker_key_counter += 1
            st.rerun()

    if start:
        lat, lon = st.session_state.selected_location
        with st.spinner("Meteoroloji verisi alınıyor..."):
            weather = fetch_weather_data(lat, lon, 2024)

        rainfall_array = None
        if weather is not None and len(weather) > 0:
            rainfall_array = weather["prcp"].to_numpy(dtype=float)
            st.session_state.weather_data = weather
            st.session_state.weather_source = "Open-Meteo 2024"
        else:
            st.session_state.weather_data = None
            st.session_state.weather_source = "Stokastik (fallback)"
            st.warning(
                "Open-Meteo verisi alınamadı. Stokastik yağışla devam ediliyor."
            )

        with st.spinner("365 günlük simülasyon çalışıyor..."):
            engine = SimulationEngine(
                roof_area=çatı_alanı,
                roof_efficiency=çatı_verimlilik,
                tank_capacity=depo_kapasitesi,
                worker_count=çalışan_sayısı,
                rain_seed=yağış_tohumu,
                external_rainfall=rainfall_array,
            )
            engine.economy.water_price = su_fiyatı / 1000  # ₺/m³ → ₺/L
            engine.economy.tank_cost = depo_maliyeti
            engine.economy.installation_cost = kurulum_maliyeti
            engine.economy.maintenance_cost_annual = bakım_maliyeti
            engine.economy.water_price_escalation = su_fiyat_artışı
            engine.economy.discount_rate = iskonto_oranı
            engine.economy.system_lifespan_years = int(sistem_ömrü)
            results = engine.run_full_simulation()

        st.session_state.simulation_engine = engine
        st.session_state.sim_results = results
        st.session_state.simulation_run = True
        st.session_state.frame_idx = 0
        st.session_state.is_playing = False
        st.session_state._last_scope = st.session_state.anim_scope
        st.session_state._last_month = int(st.session_state.anim_month)
        st.success("Simülasyon tamamlandı. Sağ taraftaki animasyonu oynatabilirsiniz.")
        st.rerun()


# --------- Sağ: PyDeck canlı overlay (izole fragment) ---------
# Fragment ile sadece sağ panel rerun eder; sol haritadaki Folium + st_folium
# widget'ı her gün yeniden oluşmaz → harita gidip gelmez.
@st.fragment
def _render_animation_panel(
    results: dict,
    roof_area: float,
    tank_capacity: float,
    rain_seed: int,
) -> None:
    daily = results["daily_history"]
    rainfall_series = np.array(daily["rainfall"], dtype=float)
    tank_levels_l = np.array(daily["tank_level"], dtype=float)
    tank_cap = float(results["system_parameters"]["tank_capacity"])
    tank_pct_series = np.clip(tank_levels_l / tank_cap * 100.0, 0, 100)
    n_days = len(rainfall_series)
    anim_dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    month_names = [
        "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
        "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
    ]

    scope_c1, scope_c2 = st.columns([1, 1])
    with scope_c1:
        scope = st.radio(
            "Zaman Ölçeği",
            options=["Aylık", "Yıllık"],
            horizontal=True,
            key="anim_scope",
        )
    with scope_c2:
        if scope == "Aylık":
            sel_month = st.selectbox(
                "Ay",
                options=list(range(1, 13)),
                format_func=lambda m: month_names[m - 1],
                key="anim_month",
            )
        else:
            sel_month = None

    scope_changed = (
        scope != st.session_state._last_scope
        or (scope == "Aylık" and sel_month != st.session_state._last_month)
    )

    if scope == "Aylık":
        mask = anim_dates.month == sel_month
        idx_pool = np.where(mask)[0]
    else:
        idx_pool = np.arange(n_days)
    if len(idx_pool) == 0:
        idx_pool = np.arange(n_days)

    if scope_changed:
        st.session_state.frame_idx = int(idx_pool[0])
        st.session_state.is_playing = False
        st.session_state._last_scope = scope
        st.session_state._last_month = int(sel_month) if sel_month else 1

    current_frame = int(st.session_state.frame_idx)
    pool_positions = np.where(idx_pool == current_frame)[0]
    if len(pool_positions) == 0:
        current_pos = 0
        st.session_state.frame_idx = int(idx_pool[0])
        current_frame = int(idx_pool[0])
    else:
        current_pos = int(pool_positions[0])

    slider_key_m = f"day_slider_m{sel_month}" if scope == "Aylık" else None
    slider_key_y = "day_slider_y"

    if scope == "Aylık":
        min_day = int(anim_dates[idx_pool[0]].day)
        max_day = int(anim_dates[idx_pool[-1]].day)
        cur_day = int(anim_dates[current_frame].day)
        target_day = max(min_day, min(max_day, cur_day))
        should_sync = (
            st.session_state.is_playing
            or slider_key_m not in st.session_state
            or scope_changed
        )
        if should_sync:
            st.session_state[slider_key_m] = target_day
    else:
        target_idx = current_frame + 1
        should_sync = (
            st.session_state.is_playing
            or slider_key_y not in st.session_state
            or scope_changed
        )
        if should_sync:
            st.session_state[slider_key_y] = target_idx

    ctrl_a, ctrl_b, ctrl_c = st.columns([3, 1, 1])
    with ctrl_a:
        if scope == "Aylık":
            slider_label = f"{month_names[sel_month - 1]} — Takvim Günü"
            picked_day = st.slider(
                slider_label,
                min_value=min_day, max_value=max_day,
                key=slider_key_m,
            )
            match = [i for i in idx_pool if int(anim_dates[i].day) == picked_day]
            if match:
                new_frame = int(match[0])
                if new_frame != current_frame:
                    st.session_state.frame_idx = new_frame
                    current_frame = new_frame
                    current_pos = int(np.where(idx_pool == new_frame)[0][0])
        else:
            picked_idx = st.slider(
                "Gün (Yıllık)",
                min_value=1, max_value=n_days,
                key=slider_key_y,
            )
            new_frame = picked_idx - 1
            if new_frame != current_frame:
                st.session_state.frame_idx = new_frame
                current_frame = new_frame
                current_pos = int(np.where(idx_pool == new_frame)[0][0])

    with ctrl_b:
        if st.button(
            "⏸ Durdur" if st.session_state.is_playing else "▶ Oynat",
            use_container_width=True,
            key="play_btn",
        ):
            if not st.session_state.is_playing and current_pos >= len(idx_pool) - 1:
                st.session_state.frame_idx = int(idx_pool[0])
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun(scope="fragment")
    with ctrl_c:
        if st.button("⏮ Başa", use_container_width=True, key="reset_frame_btn"):
            st.session_state.frame_idx = int(idx_pool[0])
            st.session_state.is_playing = False
            st.rerun(scope="fragment")

    frame_idx = int(st.session_state.frame_idx)
    rain_today = float(rainfall_series[frame_idx])
    pct_today = float(tank_pct_series[frame_idx])
    date_today = anim_dates[frame_idx]

    lat, lon = st.session_state.selected_location
    deck = build_pydeck_overlay(
        lat=lat,
        lon=lon,
        rain_mm_today=rain_today,
        tank_pct_today=pct_today,
        roof_area_m2=roof_area,
        tank_capacity_l=tank_capacity,
        frame_idx=frame_idx,
        seed=rain_seed,
    )
    st.pydeck_chart(deck, use_container_width=True)

    lg1, lg2, lg3, lg4 = st.columns(4)
    with lg1:
        if scope == "Aylık":
            st.metric(
                "Tarih",
                f"{date_today.day} {month_names[date_today.month - 1]}",
            )
        else:
            st.metric("Gün", f"{frame_idx + 1} / {n_days}")
    with lg2:
        st.metric("Yağış", f"{rain_today:.1f} mm")
    with lg3:
        st.metric("Depo", f"{pct_today:.1f}%")
    with lg4:
        st.metric("Kaynak", st.session_state.weather_source or "—")
    st.caption(
        "Bina · Depo kolonu (yükseklik = doluluk yüzdesi) · Mavi partiküller = yağmur"
    )

    if st.session_state.is_playing:
        step = 2 if scope == "Yıllık" else 1
        delay = 0.05 if scope == "Yıllık" else 0.12
        next_pos = current_pos + step
        if next_pos >= len(idx_pool):
            st.session_state.frame_idx = int(idx_pool[-1])
            st.session_state.is_playing = False
            st.rerun(scope="fragment")
        else:
            st.session_state.frame_idx = int(idx_pool[next_pos])
            time.sleep(delay)
            st.rerun(scope="fragment")


with right_col:
    st.subheader("2) Harita Üzerinde Canlı Animasyon")

    if not st.session_state.simulation_run or st.session_state.sim_results is None:
        st.info(
            "Simülasyonu başlattıktan sonra, seçili konumun üstünde günlük yağmur "
            "partikülleri ve depo seviyesi animasyonu burada görünecek."
        )
    else:
        _render_animation_panel(
            results=st.session_state.sim_results,
            roof_area=çatı_alanı,
            tank_capacity=depo_kapasitesi,
            rain_seed=yağış_tohumu,
        )


# =========================================================================
# Simülasyon sonrası sekmeler
# =========================================================================
if st.session_state.simulation_run and st.session_state.sim_results:
    results = st.session_state.sim_results
    engine = st.session_state.simulation_engine

    st.markdown("---")
    tab_overview, tab_3d, tab_worker, tab_economy, tab_export = st.tabs([
        "📊 Genel Bakış",
        "🏢 Detay 3D",
        "👥 İşçi",
        "💰 Ekonomi",
        "📥 Dışa Aktar",
    ])

    # ------------------------------------------------------------------
    # Genel Bakış
    # ------------------------------------------------------------------
    with tab_overview:
        water_m = results["water_metrics"]
        tank_m = results["tank_metrics"]
        rain_m = results["rainfall_metrics"]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Toplam Toplanan", f"{water_m['total_collected']:,.0f} L")
        with k2:
            st.metric("Toplam Tüketilen", f"{water_m['total_consumed']:,.0f} L")
        with k3:
            st.metric("Yetersiz Gün", f"{water_m['shortage_days']} gün")
        with k4:
            st.metric("Ort. Depo", f"{tank_m['avg_fill_percentage']:.1f}%")

        st.markdown("---")
        daily = results["daily_history"]
        days = daily["day"]
        levels = daily["tank_level"]
        rainfall = daily["rainfall"]

        c1, c2 = st.columns(2)
        with c1:
            fig_tank = TimeSeriesGraphs.create_tank_level_graph(
                days, levels, results["system_parameters"]["tank_capacity"]
            )
            st.plotly_chart(fig_tank, use_container_width=True)
        with c2:
            fig_rain = go.Figure()
            fig_rain.add_trace(
                go.Bar(x=days, y=rainfall, marker_color="#87CEEB", name="Yağış")
            )
            fig_rain.update_layout(
                title="Günlük Yağış (mm)",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Gün",
                yaxis_title="mm",
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        st.markdown("### 🌧️ Yağış İstatistikleri")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Toplam Yağış", f"{rain_m['total_rainfall']:.1f} mm")
        with r2:
            st.metric("Yağışlı Gün", f"{rain_m['rainy_days']} gün")
        with r3:
            st.metric("Maks. Günlük", f"{rain_m['max_daily']:.1f} mm")
        with r4:
            st.metric("Ort. Günlük", f"{rain_m['average_daily']:.2f} mm")

    # ------------------------------------------------------------------
    # Detay 3D (Plotly sahnesi)
    # ------------------------------------------------------------------
    with tab_3d:
        st.caption(
            "Ölçekli bina + depo + yağmur 3D sahnesi. Yağış verisine göre "
            "aylık ya da yıllık animasyon oluşturulur."
        )

        weather_df = st.session_state.weather_data
        sim_rainfall = np.array(results.get("daily_history", {}).get("rainfall", []))

        source_label = None
        rainfall_array = None
        dates = None
        if weather_df is not None and "prcp" in weather_df.columns and len(weather_df) > 0:
            source_label = "Open-Meteo"
            rainfall_array = weather_df["prcp"].to_numpy(dtype=float)
            dates = pd.DatetimeIndex(weather_df.index)
        elif len(sim_rainfall) > 0:
            source_label = "Stokastik"
            rainfall_array = sim_rainfall.astype(float)
            dates = pd.date_range("2024-01-01", periods=len(rainfall_array), freq="D")

        if rainfall_array is None or len(rainfall_array) == 0:
            st.warning("Animasyon için yağış verisi yok.")
        else:
            st.info(f"Veri kaynağı: **{source_label}** · {len(rainfall_array)} gün")

            month_names = [
                "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
            ]
            c1, c2 = st.columns([1, 1])
            with c1:
                scale = st.radio(
                    "Zaman Ölçeği",
                    options=["Aylık", "Yıllık"],
                    horizontal=True,
                    key="anim_scale",
                )
            with c2:
                if scale == "Aylık":
                    selected_month = st.selectbox(
                        "Ay Seç",
                        options=list(range(1, 13)),
                        format_func=lambda m: month_names[m - 1],
                        key="anim_month_tab3",
                    )
                else:
                    selected_month = None

            sysp = results["system_parameters"]
            roof_area_val = float(sysp["roof_area_m2"])
            roof_eff = float(sysp["roof_efficiency"])
            tank_capacity_val = float(sysp["tank_capacity"])
            daily_consumption = float(water_m["daily_average_consumed"])

            replay = replay_tank_for_rainfall(
                rainfall_mm=rainfall_array,
                roof_area=roof_area_val,
                efficiency=roof_eff,
                tank_capacity=tank_capacity_val,
                daily_consumption_liters=daily_consumption,
            )

            if scale == "Aylık":
                mask = dates.month == selected_month
                rain_slice = rainfall_array[mask]
                pct_slice = replay["levels_pct"][mask]
                labels = [d.strftime("%d %b") for d, keep in zip(dates, mask) if keep]
                frame_dur = 300
                title = f"{month_names[selected_month - 1]} - Yağmur & Depo"
            else:
                stride = 2 if len(rainfall_array) > 200 else 1
                rain_slice = rainfall_array[::stride]
                pct_slice = replay["levels_pct"][::stride]
                labels = [d.strftime("%d %b") for d in dates[::stride]]
                frame_dur = 120
                title = "Yıllık Yağmur & Depo"

            if len(rain_slice) == 0:
                st.warning("Seçili periyotta gösterilecek veri bulunamadı.")
            else:
                scene = Scene3D()
                anim_fig = scene.create_animated_scene(
                    daily_rain=rain_slice,
                    tank_pcts=pct_slice,
                    labels=labels,
                    roof_area=roof_area_val,
                    tank_capacity=tank_capacity_val,
                    title=title,
                    frame_duration_ms=frame_dur,
                )
                st.plotly_chart(anim_fig, use_container_width=True)

    # ------------------------------------------------------------------
    # İşçi
    # ------------------------------------------------------------------
    with tab_worker:
        worker_count = results["system_parameters"]["worker_count"]
        daily_consumed = water_m["daily_average_consumed"]
        daily_available = water_m["daily_average_collected"]

        fig_workers = draw_worker_simulation(
            worker_count, daily_available, daily_consumed
        )
        st.pyplot(fig_workers, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam İşçi", f"{worker_count} kişi")
        with col2:
            per_worker = daily_consumed / worker_count if worker_count else 0
            st.metric("Günlük/Kişi", f"{per_worker:.1f} L")
        with col3:
            satisfaction = (
                daily_available / daily_consumed * 100 if daily_consumed > 0 else 0
            )
            st.metric("Talep Karşılama", f"{min(satisfaction, 100):.1f}%")

    # ------------------------------------------------------------------
    # Ekonomi
    # ------------------------------------------------------------------
    with tab_economy:
        economic = results["economic_metrics"]
        financial = economic["financial"]
        breakeven = economic["breakeven"]
        water_e = economic["water_metrics"]

        st.subheader("Mali Özet")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Tasarruf (₺)", f"₺ {financial['cost_saved']:,.0f}")
        with e2:
            st.metric("Toplam Maliyet", f"₺ {financial['total_investment']:,.0f}")
        with e3:
            st.metric("Net Fayda", f"₺ {financial['net_benefit']:,.0f}")
        with e4:
            st.metric("ROI", f"{financial['roi_percentage']:.1f}%")

        st.markdown("---")
        p1, p2, p3 = st.columns(3)
        payback_disc = financial.get("payback_years_discounted")
        payback_simple = financial.get("payback_years_simple")
        assumptions = financial.get("assumptions", {})
        cash_flow_rows = financial.get("cash_flow_table", [])

        with p1:
            m_col, pop_col = st.columns([3, 1])
            with m_col:
                st.metric(
                    "Geri Ödeme",
                    f"{payback_disc:.1f} yıl" if payback_disc else "Ömür içinde yok",
                    help="İskonto edilmiş geri ödeme süresi (Discounted Payback).",
                )
            with pop_col:
                with st.popover("ℹ️", use_container_width=True):
                    e_pct = assumptions.get("water_price_escalation", 0) * 100
                    r_pct = assumptions.get("discount_rate", 0) * 100
                    life = assumptions.get("system_lifespan_years", 0)
                    I = assumptions.get("initial_investment", 0)
                    M = assumptions.get("maintenance_annual", 0)
                    S1 = assumptions.get("water_savings_y1", 0)
                    npv_val = financial.get("npv", 0)
                    simple_txt = (
                        f"{payback_simple:.1f} yıl" if payback_simple else "gerçekleşmez"
                    )
                    disc_txt = (
                        f"{payback_disc:.1f} yıl" if payback_disc else f"{life}+ yıl (ömür içinde yok)"
                    )
                    st.markdown("### Geri Ödeme Süresi — Metodoloji")
                    st.markdown(
                        "Basit `toplam yatırım / yıllık tasarruf` formülü, paranın "
                        "zaman değerini ve bakım giderini yok saydığı için gerçekçi "
                        "değildir. Bu simülatör **Discounted Payback** yöntemini kullanır."
                    )
                    st.markdown(
                        "**Formül**\n\n"
                        "- Yıllık kaba tasarruf: `S_t = S₁ × (1 + e)^(t−1)`\n"
                        "- Net nakit akışı: `CF_t = S_t − M`\n"
                        "- İskonto: `DCF_t = CF_t / (1 + r)^t`\n"
                        "- Geri ödeme: en küçük `n` için `Σ DCF_t ≥ I`"
                    )
                    st.markdown("**Varsayımlar**")
                    st.markdown(
                        f"- Su fiyatı reel artışı **e = {e_pct:.1f}% / yıl**\n"
                        f"- Reel iskonto oranı **r = {r_pct:.1f}% / yıl**\n"
                        f"- Sistem ömrü **{life} yıl**\n"
                        f"- İlk yatırım **I = ₺ {I:,.0f}** (kurulum + depo)\n"
                        f"- Yıllık bakım **M = ₺ {M:,.0f}**\n"
                        f"- Yıl-1 tasarruf **S₁ = ₺ {S1:,.0f}**"
                    )
                    st.markdown(
                        "**Sonuç**\n\n"
                        f"- Basit geri ödeme (iskontosuz): **{simple_txt}**\n"
                        f"- İskonto edilmiş geri ödeme: **{disc_txt}**  ← metrikte gösterilen\n"
                        f"- NPV ({life} yıl): **₺ {npv_val:,.0f}**"
                    )
                    if cash_flow_rows:
                        st.markdown("**Yıl-yıl nakit akışı (₺)**")
                        cf_df = pd.DataFrame(cash_flow_rows).round(0)
                        cf_df = cf_df.rename(columns={
                            "year": "Yıl",
                            "savings": "Tasarruf",
                            "maintenance": "Bakım",
                            "net_cf": "Net Nakit",
                            "dcf": "İskontolu",
                            "cum_dcf": "Kümülatif",
                        })
                        st.dataframe(cf_df, hide_index=True, use_container_width=True)
        with p2:
            st.metric(
                "Ekonomik Uygun",
                "✓ Evet" if payback_disc is not None else "✗ Hayır",
            )
        with p3:
            st.metric(
                "Yıllık Tasarruf",
                f"₺ {financial.get('annual_savings', 0):,.0f}",
            )

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Maliyet Dağılımı")
            install_display = financial.get(
                "installation_cost",
                getattr(engine.economy, "installation_cost", 0),
            )
            tank_cost_display = financial.get(
                "tank_cost", getattr(engine.economy, "tank_cost", 0)
            )
            maintenance_cost_display = financial.get(
                "maintenance_annual",
                getattr(engine.economy, "maintenance_cost_annual", 0),
            )
            initial_investment_display = financial.get(
                "initial_investment", install_display + tank_cost_display
            )
            cost_data = {
                "Kurulum": f"₺ {install_display:,.0f}",
                "Depo": f"₺ {tank_cost_display:,.0f}",
                "İlk Yatırım (I)": f"₺ {initial_investment_display:,.0f}",
                "Yıllık Bakım (M)": f"₺ {maintenance_cost_display:,.0f}",
            }
            for label, value in cost_data.items():
                st.write(f"• {label}: **{value}**")
        with col_b:
            st.subheader("Su ve Tasarruf")
            data = {
                "Toplanan Su": f"{water_e['collected_liters']:,.0f} L",
                "Tüketilen Su": f"{water_e['consumed_liters']:,.0f} L",
                "Kullanım": f"{water_e['utilization_rate']:.1f}%",
                "Tasarruf": f"₺ {financial['cost_saved']:,.0f}",
            }
            for label, value in data.items():
                st.write(f"• {label}: **{value}**")

    # ------------------------------------------------------------------
    # Dışa Aktar
    # ------------------------------------------------------------------
    with tab_export:
        df_history = pd.DataFrame(results["daily_history"])
        st.dataframe(df_history, use_container_width=True, height=320)

        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Günlük Verileri İndir (CSV)",
            data=csv,
            file_name=f"simulation_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Simülasyon Sonuçları (JSON)",
            data=json_data,
            file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

else:
    st.markdown("---")
    with st.expander("📋 Bu Simülasyon Hakkında", expanded=False):
        st.markdown(
            """
            Bu uygulama yağmur hasadı sistemlerini etkileşimli olarak modellemek için hazırlandı.

            - **Konum**: Haritaya ölçekli bina modelini sürükleyip bırakın veya enlem/boylam girin.
            - **Meteoroloji**: Open-Meteo Archive API'dan günlük yağış verisi çekilir.
            - **Simülasyon**: 365 günlük yağış, depo, tüketim ve ekonomik analizi çalıştırır.
            - **Animasyon**: Harita üzerinde PyDeck ile günlük yağmur partikülleri + depo
              kolonu + 3D bina extruzyonu gösterilir.
            - **Fallback**: Gerçek veri alınamazsa stokastik model devreye girer.
            """
        )


# =========================================================================
# Altbilgi
# =========================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6B7280;font-size:0.85rem;'>"
    "🌧️ Yağmur Hasadı Simülasyon Platformu · Open-Meteo · PyDeck + Folium"
    "</div>",
    unsafe_allow_html=True,
)
