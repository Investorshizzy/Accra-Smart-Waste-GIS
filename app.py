import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
import math

st.set_page_config(
    page_title="Accra Smart Waste GIS",
    page_icon="🚛",
    layout="wide"
)

# ---------------------------------------------------------
# 1. ACCRA GEOSPATIAL SEED DATA
# ---------------------------------------------------------
DEPOT = {
    "name": "Mudor Waste Transfer Station (Depot)",
    "lat": 5.5342,
    "lon": -0.2185
}

BINS_SEED = [
    {"id": "BIN-01", "name": "Makola Market Central", "lat": 5.5458, "lon": -0.2078, "market_factor": 1.9, "capacity_liters": 10000},
    {"id": "BIN-02", "name": "Agbogbloshie Commercial", "lat": 5.5510, "lon": -0.2245, "market_factor": 2.2, "capacity_liters": 10000},
    {"id": "BIN-03", "name": "Kaneshie Market Complex", "lat": 5.5654, "lon": -0.2372, "market_factor": 1.7, "capacity_liters": 10000},
    {"id": "BIN-04", "name": "Circle Kwame Nkrumah Overpass", "lat": 5.5583, "lon": -0.2075, "market_factor": 1.5, "capacity_liters": 5000},
    {"id": "BIN-05", "name": "Osu Oxford Street", "lat": 5.5560, "lon": -0.1834, "market_factor": 1.2, "capacity_liters": 5000},
    {"id": "BIN-06", "name": "Adabraka Polyclinic Area", "lat": 5.5592, "lon": -0.2132, "market_factor": 1.1, "capacity_liters": 5000},
    {"id": "BIN-07", "name": "Labadi Beach Road", "lat": 5.5611, "lon": -0.1558, "market_factor": 1.3, "capacity_liters": 5000},
    {"id": "BIN-08", "name": "Ridge Hospital Environs", "lat": 5.5628, "lon": -0.1989, "market_factor": 0.9, "capacity_liters": 5000},
    {"id": "BIN-09", "name": "Jamestown Lighthouse", "lat": 5.5349, "lon": -0.2120, "market_factor": 1.0, "capacity_liters": 5000},
    {"id": "BIN-10", "name": "Korle Bu Outpatients", "lat": 5.5398, "lon": -0.2301, "market_factor": 1.4, "capacity_liters": 10000},
]

# ---------------------------------------------------------
# 2. MACHINE LEARNING ENGINE: FILL-LEVEL PREDICTOR
# ---------------------------------------------------------
@st.cache_resource
def train_fill_prediction_model():
    """Trains a Random Forest Regressor on simulated Accra telemetry data."""
    np.random.seed(42)
    n_samples = 2500
    
    # Feature columns:
    # [current_fill_pct, hours_since_last_pickup, market_activity_multiplier, is_weekend, rainfall_mm]
    current_fill = np.random.uniform(10, 95, n_samples)
    hours_elapsed = np.random.uniform(1, 36, n_samples)
    market_factor = np.random.uniform(0.8, 2.5, n_samples)
    is_weekend = np.random.choice([0, 1], n_samples)
    rainfall_mm = np.random.exponential(scale=5.0, size=n_samples)
    
    # Target: Fill percentage 12 hours from now
    fill_rate_hourly = (market_factor * 1.8) + (is_weekend * 0.8) + (rainfall_mm * 0.15)
    future_fill = current_fill + (fill_rate_hourly * 12) + np.random.normal(0, 3, n_samples)
    future_fill = np.clip(future_fill, 0, 100)
    
    X = np.column_stack([current_fill, hours_elapsed, market_factor, is_weekend, rainfall_mm])
    y = future_fill
    
    model = RandomForestRegressor(n_estimators=70, random_state=42)
    model.fit(X, y)
    return model

# ---------------------------------------------------------
# 3. ROUTE OPTIMIZATION ALGORITHM (Nearest Neighbor TSP)
# ---------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def solve_tsp_nearest_neighbor(depot, target_bins):
    """Generates an ordered collection sequence starting and ending at Mudor Depot."""
    if not target_bins:
        return [], 0.0

    unvisited = target_bins.copy()
    current_pt = depot
    route = [depot]
    total_dist = 0.0

    while unvisited:
        nearest_idx = None
        min_d = float('inf')
        for idx, b in enumerate(unvisited):
            d = haversine_km(current_pt["lat"], current_pt["lon"], b["lat"], b["lon"])
            if d < min_d:
                min_d = d
                nearest_idx = idx

        next_bin = unvisited.pop(nearest_idx)
        route.append(next_bin)
        total_dist += min_d
        current_pt = next_bin

    # Return to depot
    return_dist = haversine_km(current_pt["lat"], current_pt["lon"], depot["lat"], depot["lon"])
    total_dist += return_dist
    route.append(depot)
    return route, round(total_dist, 2)

# ---------------------------------------------------------
# 4. STREAMLIT APP LAYOUT
# ---------------------------------------------------------
st.title("Smart Waste Management & GIS - Accra Metro")
st.markdown("Real-time monitoring, fill-level machine learning prediction, and route optimization across Accra.")

model = train_fill_prediction_model()

# Sidebar: Controls
st.sidebar.header("Operational Parameters")
overflow_threshold = st.sidebar.slider("Urgent Collection Threshold (%)", min_value=50, max_value=95, value=75, step=5)
sim_rainfall = st.sidebar.slider("Forecasted Rainfall in Accra (mm)", min_value=0.0, max_value=40.0, value=8.0, step=1.0)
sim_is_weekend = st.sidebar.checkbox("Is Weekend / Heavy Market Day?", value=True)

# Generate current bin states
np.random.seed(101)
bins_data = []
for b in BINS_SEED:
    curr_fill = np.random.uniform(30, 88)
    hours_elapsed = np.random.uniform(4, 28)
    features = np.array([[curr_fill, hours_elapsed, b["market_factor"], int(sim_is_weekend), sim_rainfall]])
    predicted_fill = float(model.predict(features)[0])
    
    bins_data.append({
        **b,
        "current_fill": round(curr_fill, 1),
        "predicted_fill_12h": round(predicted_fill, 1),
        "needs_collection": predicted_fill >= overflow_threshold
    })

df_bins = pd.DataFrame(bins_data)

# KPI Metrics Bar
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
critical_count = len(df_bins[df_bins["needs_collection"]])
avg_fill = round(df_bins["current_fill"].mean(), 1)
avg_pred_fill = round(df_bins["predicted_fill_12h"].mean(), 1)

kpi1.metric("Total Monitored Skips", len(df_bins))
kpi2.metric("Average Current Fill", f"{avg_fill}%")
kpi3.metric("Avg Predicted Fill (12h)", f"{avg_pred_fill}%")
kpi4.metric("Action Required (>Threshold)", f"{critical_count} Skips", delta_color="inverse")

# Route Calculation
critical_bins = [b for b in bins_data if b["needs_collection"]]
ordered_route, total_km = solve_tsp_nearest_neighbor(DEPOT, critical_bins)

# ---------------------------------------------------------
# 5. WEB GIS INTERACTION (FOLIUM)
# ---------------------------------------------------------
col_map, col_details = st.columns([7, 3])

with col_map:
    st.subheader("Accra Spatial Monitoring & Dynamic Routing")
    
    # Center map on Accra
    carto_key = st.secrets.get("CARTO_API_KEY", "cb1_4049_1_baaa2180ec1448f2bb56f23c")
    tiles_url = f"https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png?key={carto_key}"
   attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(location=[5.5600, -0.2050], zoom_start=13, tiles=tiles_url, attr=attr)

    # Add Central Depot
    folium.Marker(
        location=[DEPOT["lat"], DEPOT["lon"]],
        tooltip="START/END: " + DEPOT["name"],
        icon=folium.Icon(color="black", icon="industry", prefix="fa")
    ).add_to(m)

    # Plot Bins
    for b in bins_data:
        # Determine status color
        if b["predicted_fill_12h"] >= overflow_threshold:
            color = "red"
            status = "CRITICAL (Scheduled)"
        elif b["predicted_fill_12h"] >= 60:
            color = "orange"
            status = "WARNING"
        else:
            color = "green"
            status = "NORMAL"

        popup_html = f"""
        <b>{b['name']}</b> ({b['id']})<br>
        Capacity: {b['capacity_liters']:,} L<br>
        Current Fill: <b>{b['current_fill']}%</b><br>
        12h ML Prediction: <b>{b['predicted_fill_12h']}%</b><br>
        Status: <span style='color:{color}; font-weight:bold;'>{status}</span>
        """
        
        folium.CircleMarker(
            location=[b["lat"], b["lon"]],
            radius=9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{b['name']}: {b['predicted_fill_12h']}% (12h)"
        ).add_to(m)

    # Draw Route Polyline if critical stops exist
    if len(ordered_route) > 2:
        route_coords = [[stop["lat"], stop["lon"]] for stop in ordered_route]
        folium.PolyLine(
            locations=route_coords,
            color="#0066cc",
            weight=4,
            opacity=0.8,
            dash_array="6",
            tooltip=f"Optimized Path: {total_km} km"
        ).add_to(m)

    st_folium(m, width="100%", height=550)

with col_details:
    st.subheader("Dispatch Sequence")
    if critical_bins:
        st.write(f"**Total Distance:** `{total_km} km`")
        for i, stop in enumerate(ordered_route):
            icon = "🏁" if i in [0, len(ordered_route) - 1] else "📦"
            st.write(f"{icon} **Leg {i}:** {stop['name']}")
    else:
        st.success("No skips have crossed the critical threshold. No truck dispatch required.")

# Data Table Display
with st.expander("View Full Geospatial Sensor & ML Predictions Table"):
    st.dataframe(
        df_bins[["id", "name", "capacity_liters", "current_fill", "predicted_fill_12h", "needs_collection"]],
        use_container_width=True
    )
