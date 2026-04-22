# 🏗️ PRODUCTION-GRADE GEOSPATIAL RAINWATER HARVESTING PLATFORM
## Enterprise Architecture & Implementation Roadmap

**Version:** 2.0  
**Status:** Design Phase  
**Author:** Senior Software Architect + Data Science Lead  
**Last Updated:** April 2026

---

## 📋 TABLE OF CONTENTS
1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Module Structure (Production-Ready)](#module-structure)
4. [Data Pipeline](#data-pipeline)
5. [Model Pipeline](#model-pipeline)
6. [API Integration Design](#api-integration)
7. [Simulation Engine (Enhanced)](#simulation-engine)
8. [UI/UX Component Architecture](#ui-architecture)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Scalability & DevOps](#scalability)
11. [Risk Analysis & Mitigation](#risk-analysis)

---

## 🎯 EXECUTIVE SUMMARY

This platform evolves from a local simulation engine into a **cloud-ready, multi-tenant geospatial analytics engine** that:

- **Ingests**: Real weather data from 10+ years of historical records
- **Models**: Stochastic rainfall + climate scenarios + building characteristics
- **Simulates**: 365+ day scenarios with full uncertainty quantification
- **Visualizes**: Interactive 2D/3D maps with real-time analytics
- **Scales**: From single buildings to city/region portfolios
- **Optimizes**: Water yield, tank capacity, and ROI simultaneously

**Key Shift**: From isolated simulation → Integrated **Data → Model → Insight** pipeline.

---

## 🏛️ SYSTEM ARCHITECTURE OVERVIEW

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       STREAMLIT FRONTEND                         │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  Map Interface       │  │  Dashboard & Analytics          │ │
│  │  - Leaflet/Folium    │  │  - Time Series Charts           │ │
│  │  - Click Location    │  │  - KPI Cards                    │ │
│  │  - Geolocation API   │  │  - Export Panels                │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  3D Visualization    │  │  Scenario Builder               │ │
│  │  - Pydeck/Deck.gl    │  │  - Climate Sliders              │ │
│  │  - Building Layers   │  │  - Parameter Grid               │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (REST/WebSocket API)
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Router / GraphQL Endpoint                      │   │
│  │  - Request Validation & Auth                            │   │
│  │  - Response Serialization                               │   │
│  │  - Caching (Redis)                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Data Orchestration Services                             │   │
│  │  - WeatherService (Meteostat API Client)                │   │
│  │  - LocationService (Geocoding, Validation)               │   │
│  │  - BuildingService (Portfolio Management)                │   │
│  │  - SimulationService (Engine Orchestrator)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Modeling Services                                       │   │
│  │  - RainfallModel (Statistical Distribution)              │   │
│  │  - HarvestingModel (Engineering Calculations)            │   │
│  │  - StorageOptimizer (Tank Sizing)                        │   │
│  │  - EconomicsAnalyzer (ROI, Payback)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Simulation Services                                     │   │
│  │  - ScenarioGenerator (Climate, Extreme Events)           │   │
│  │  - SimulationEngine (Main Orchestrator)                  │   │
│  │  - UncertaintyQuantification (Monte Carlo, Sensitivity) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  ┌───────────────────────┐  ┌──────────────────────────────┐    │
│  │  TimeSeries DB        │  │  Cache Layer                 │    │
│  │  - InfluxDB / Postgres│  │  - Redis (Query Results)     │    │
│  │  - Weather Data       │  │  - Memcached (Session Data)  │    │
│  │  - Simulation Results │  └──────────────────────────────┘    │
│  └───────────────────────┘  ┌──────────────────────────────┐    │
│  ┌───────────────────────┐  │  Geospatial DB               │    │
│  │  Document Store       │  │  - PostGIS (Spatial Index)   │    │
│  │  - MongoDB / Firestore│  │  - Building Geometries       │    │
│  │  - Portfolio Data     │  │  - Administrative Boundaries │    │
│  └───────────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Meteostat   │  │ OpenStreetMap│  │ Cloud Storage (S3)     │  │
│  │ API         │  │ / Overpass   │  │ Data Export             │  │
│  └─────────────┘  │ API          │  └────────────────────────┘  │
│                   └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 MODULE STRUCTURE (Production-Ready)

### Directory Tree

```
rainwater-platform/
│
├── frontend/                          # Streamlit + React Components
│   ├── streamlit_app.py               # Main entry point
│   ├── pages/                         # Multi-page structure
│   │   ├── 1_🗺️_Location_Selection.py
│   │   ├── 2_📊_Data_Explorer.py
│   │   ├── 3_🔬_Model_Analysis.py
│   │   ├── 4_⚙️_Scenario_Builder.py
│   │   ├── 5_🏢_Building_Portfolio.py
│   │   └── 6_💰_Economics.py
│   ├── components/                    # Reusable UI components
│   │   ├── map_widget.py              # Folium/Leaflet wrapper
│   │   ├── 3d_visualizer.py           # Pydeck visualization
│   │   ├── chart_builder.py           # Plotly dashboard
│   │   ├── sidebar_controls.py        # Parameter interface
│   │   └── data_grid.py               # AG Grid wrapper
│   └── utils/
│       ├── session_manager.py         # State management
│       └── theme_config.py            # UI styling
│
├── backend/                           # FastAPI / Service Layer
│   ├── main.py                        # API entry point
│   ├── routers/                       # Route handlers
│   │   ├── location.py                # Location endpoints
│   │   ├── weather.py                 # Weather data endpoints
│   │   ├── simulation.py              # Simulation control
│   │   ├── buildings.py               # Building CRUD
│   │   ├── scenarios.py               # Scenario management
│   │   └── analytics.py               # Analytics queries
│   ├── services/                      # Business Logic Layer
│   │   ├── weather_service.py         # Meteostat integration
│   │   ├── location_service.py        # Geocoding + validation
│   │   ├── building_service.py        # Building portfolio
│   │   ├── simulation_service.py      # Simulation orchestration
│   │   ├── analytics_service.py       # Report generation
│   │   └── cache_service.py           # Redis wrapper
│   ├── models/                        # Pydantic data models
│   │   ├── location.py                # LocationInput, Location
│   │   ├── weather.py                 # WeatherData, DailyPrecip
│   │   ├── simulation.py              # SimParams, SimResult
│   │   ├── building.py                # Building, Portfolio
│   │   └── scenario.py                # ClimateScenario, Sensitivity
│   ├── middleware/                    # Cross-cutting concerns
│   │   ├── auth.py                    # Token validation
│   │   ├── logging.py                 # Structured logging
│   │   ├── error_handler.py           # Error standardization
│   │   └── rate_limiter.py            # Quota management
│   └── config/
│       ├── settings.py                # Env-based config
│       └── database.py                # DB connections
│
├── core/                              # Core Simulation & Modeling
│   ├── __init__.py
│   ├── rainfall_model/
│   │   ├── distribution_fitter.py     # MLE parameter fitting
│   │   ├── stochastic_generator.py    # Rainfall series generation
│   │   ├── seasonal_model.py          # Monthly decomposition
│   │   ├── validation.py              # Goodness-of-fit tests
│   │   └── climate_scenarios.py       # Future projections
│   │
│   ├── harvesting_model/
│   │   ├── collection.py              # V = P × A × C formula
│   │   ├── tank_dynamics.py           # Storage state evolution
│   │   ├── optimization.py            # Tank sizing + efficiency
│   │   └── yield_calculator.py        # Daily/Annual aggregation
│   │
│   ├── simulation/
│   │   ├── engine.py                  # Main simulation loop
│   │   ├── scenario.py                # Scenario execution
│   │   ├── uncertainty.py             # Monte Carlo, Sensitivity
│   │   ├── parallel_runner.py         # Multi-scenario execution
│   │   └── result_aggregator.py       # Result compilation
│   │
│   ├── analytics/
│   │   ├── metrics.py                 # KPI calculations
│   │   ├── statistics.py              # Aggregate statistics
│   │   ├── timeseries.py              # Time-series decomposition
│   │   └── extremes.py                # Extreme event analysis
│   │
│   └── utils/
│       ├── constants.py               # Global constants
│       ├── validators.py              # Input validation
│       ├── datetime_utils.py          # Temporal utilities
│       └── interpolation.py           # Missing data handling
│
├── data/                              # Data Management Layer
│   ├── database/
│   │   ├── models.py                  # SQLAlchemy ORM models
│   │   ├── schemas.py                 # Schema definitions
│   │   ├── migrations/                # Alembic migrations
│   │   └── seeds/                     # Initial data
│   │
│   ├── loaders/
│   │   ├── meteostat_loader.py        # Meteostat API client
│   │   ├── osm_loader.py              # OSM/Overpass integration
│   │   ├── csv_importer.py            # Batch data import
│   │   └── cache_manager.py           # Caching strategy
│   │
│   ├── processors/
│   │   ├── cleaning.py                # Data quality checks
│   │   ├── interpolation.py           # Missing value handling
│   │   ├── normalization.py           # Feature scaling
│   │   └── aggregator.py              # Temporal aggregation
│   │
│   └── exporters/
│       ├── csv_exporter.py            # CSV output
│       ├── json_exporter.py           # JSON serialization
│       ├── geojson_exporter.py        # Spatial features
│       └── report_generator.py        # PDF/HTML reports
│
├── tests/                             # Comprehensive Testing
│   ├── unit/
│   │   ├── test_rainfall_model.py
│   │   ├── test_harvesting_model.py
│   │   ├── test_simulation.py
│   │   ├── test_analytics.py
│   │   └── test_validators.py
│   │
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_simulation_pipeline.py
│   │   ├── test_api_endpoints.py
│   │   └── test_database.py
│   │
│   ├── fixtures/
│   │   ├── conftest.py                # pytest fixtures
│   │   ├── mock_weather_data.py       # Test data
│   │   └── sample_buildings.py
│   │
│   └── performance/
│       ├── benchmark_simulation.py
│       └── load_test_api.py
│
├── deployment/                        # DevOps & Infrastructure
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.frontend
│   │   └── docker-compose.yml
│   │
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   └── configmap.yaml
│   │
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── rds.tf
│   │   ├── redis.tf
│   │   └── s3.tf
│   │
│   ├── ci_cd/
│   │   ├── .github/workflows/
│   │   │   ├── test.yml
│   │   │   ├── lint.yml
│   │   │   └── deploy.yml
│   │   └── scripts/
│   │       ├── lint.sh
│   │       ├── test.sh
│   │       └── deploy.sh
│   │
│   └── monitoring/
│       ├── prometheus.yml
│       ├── grafana_dashboards/
│       └── alert_rules.yml
│
├── docs/                              # Documentation
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   ├── DATA_DICTIONARY.md
│   └── images/
│
├── config/
│   ├── settings.yaml                  # Environment config
│   ├── database.yaml                  # DB schemas
│   └── logging.yaml                   # Logging setup
│
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Modern packaging
├── Makefile                           # Development shortcuts
├── .env.example                       # Environment template
├── .dockerignore
├── .gitignore
├── README.md
└── ARCHITECTURE.md
```

---

## 🔄 DATA PIPELINE (DETAILED)

### Stage 1: Data Ingestion

```python
DATA SOURCES
├── Meteostat API (Primary)
│   ├── Daily precipitation (mm)
│   ├── Temperature (°C)
│   ├── Relative humidity (%)
│   └── Wind speed (km/h)
│   └── [CACHE: PostgreSQL + Redis]
│
├── OpenStreetMap / Overpass API
│   ├── Building footprints (geometries)
│   ├── Roof types & materials
│   ├── Building height estimates
│   └── [CACHE: PostGIS database]
│
└── User-Provided Data
    ├── Manual roof area input
    ├── Building characteristics
    ├── Tank capacity specs
    └── [STORE: Document DB]

QUALITY ASSURANCE LAYER
├── Validation checks
│   ├── Range checks (0 ≤ precip ≤ 500 mm)
│   ├── Temporal consistency (no gaps > 7 days)
│   ├── Geospatial validation (lat/lon bounds)
│   └── Coordinate projection verification
│
├── Missing data handling
│   ├── Linear interpolation (gaps < 3 days)
│   ├── Seasonal average fill (gaps 3-30 days)
│   ├── Multiple imputation (gaps > 30 days)
│   └── Flag anomalous records
│
└── Data profiling
    ├── Distribution tests (Shapiro-Wilk)
    ├── Outlier detection (IQR, Isolation Forest)
    ├── Autocorrelation analysis
    └── Seasonal decomposition
```

### Stage 2: Data Storage

```
PostgreSQL          InfluxDB           MongoDB           PostGIS
├── Weather         ├── Time series    ├── Portfolio     ├── Building
│  metadata         │  data             │  profiles        footprints
├── Location        ├── Simulation     ├── User           ├── Admin
│  cache            │  results          │  settings         boundaries
├── User            └── Metrics         └── Scenarios    └── Spatial
│  accounts                                                indexes

                    (All replicated to S3 for backup)
```

### Stage 3: Data Processing Pipeline

```python
class DataPipeline:
    """
    Orchestrates data from source → model-ready format
    """
    
    async def fetch_weather(location, start_year, end_year):
        """
        1. Query Meteostat API with fallback to local cache
        2. Validate & clean raw records
        3. Aggregate to daily level
        4. Fill gaps via interpolation
        5. Cache in InfluxDB + PostgreSQL
        6. Return processed DataFrame
        """
        pass
    
    async def fetch_buildings(lat, lon, radius_km=1):
        """
        1. Query OSM Overpass API for geometries
        2. Extract roof area via polygon area calculation
        3. Estimate height from OSM tags + DEM
        4. Standardize schema
        5. Cache in PostGIS
        6. Return GeoDataFrame
        """
        pass
    
    async def harmonize_temporal(df_weather):
        """
        Ensure 365/366 days with no gaps:
        1. Check completeness
        2. Interpolate missing values
        3. Handle leap years
        4. Return ready-for-modeling DataFrame
        """
        pass
```

---

## 📈 MODEL PIPELINE (DETAILED)

### Stage 1: Statistical Rainfall Model

#### 1.1 Parameter Fitting (Offline, Pre-Computed)

```python
class RainfallModelFitter:
    """
    Fit Bernoulli + Gamma distribution to historical data.
    Run ONCE per location, cache results.
    """
    
    def __init__(self, precip_data):
        self.data = precip_data  # 10+ years daily rainfall (mm)
    
    def fit_bernoulli(self):
        """
        Estimate rain probability: p = (rainy_days / total_days)
        - Rainy day = precip > threshold (e.g., 0.1 mm)
        - Output: scalar p ∈ [0, 1]
        - Seasonal variant: 12 different p values (Jan-Dec)
        """
        pass
    
    def fit_gamma(self):
        """
        Estimate gamma shape (α) & scale (β) for rainy days only.
        Use Maximum Likelihood Estimation (scipy.stats.gamma.fit).
        - Input: precip on rainy days only
        - Output: (shape, loc, scale)
        - Seasonal: 12 different parameter sets
        """
        pass
    
    def validate_fit(self, n_bootstrap=1000):
        """
        Goodness-of-fit tests:
        - Kolmogorov-Smirnov test (synthetic vs empirical)
        - Anderson-Darling test
        - Visual Q-Q plots
        Output: p-values, diagnostic plots
        """
        pass
```

#### 1.2 Synthetic Rainfall Generation

```python
class StochasticRainfallGenerator:
    """
    Generate synthetic daily rainfall time-series.
    Reproduces historical statistics + enables future projections.
    """
    
    def generate_series(self, n_days=365, seed=42, seasonal=True):
        """
        Algorithm:
        1. For each day:
           a. Sample from Bernoulli(p) → is_rainy?
           b. If rainy: sample from Gamma(α, β)
           c. If not rainy: P_t = 0
        2. Return 1D array of n_days precipitation values
        
        Inputs:
        - n_days: simulation horizon (typically 365)
        - seed: for reproducibility
        - seasonal: use monthly parameters?
        
        Output: np.array([365]) daily rainfall in mm
        """
        np.random.seed(seed)
        rainfall = np.zeros(n_days)
        for day in range(n_days):
            month = get_month(day)
            p_rain = self.p_rain[month] if seasonal else self.p_rain_annual
            if np.random.rand() < p_rain:
                alpha, beta = self.gamma_params[month] if seasonal else self.gamma_params_annual
                rainfall[day] = np.random.gamma(alpha, beta)
        return rainfall
```

#### 1.3 Climate Scenario Adjustment

```python
class ClimateScenarioModifier:
    """
    Adjust rainfall parameters for climate scenarios.
    Implements IPCC-style climate change projections.
    """
    
    def apply_scenario(self, base_params, scenario='moderate'):
        """
        Scenarios (tied to CMIP6/SSP projections):
        
        Scenario         | ΔFrequency | ΔIntensity | Description
        ─────────────────┼────────────┼────────────┼──────────────────────
        optimistic       | -5%        | -5%        | Reduced rainfall
        moderate         | -10%       | +15%       | More intense, less frequent
        pessimistic      | -20%       | +30%       | Severe pattern change
        extreme          | -15%       | +50%       | Extreme intensities
        
        Implementation:
        - Multiply p_rain by (1 + ΔFrequency/100)
        - Multiply gamma.scale by (1 + ΔIntensity/100)
        - Preserve shape parameter
        """
        pass
```

### Stage 2: Rainwater Harvesting Model

#### 2.1 Collection & Tank Dynamics

```python
class HarvestingModel:
    """
    Core engineering model: V = P × A × C
    Plus tank state evolution with constraints.
    """
    
    def __init__(self, roof_area, efficiency, tank_capacity, tank_initial_level):
        self.A = roof_area              # m²
        self.C = efficiency             # dimensionless (0-1)
        self.V_max = tank_capacity      # liters
        self.V_current = tank_initial_level * V_max  # liters
    
    def daily_collection(self, precip_mm):
        """
        Calculate potential water collected on a given day.
        
        Formula: V_collected = P × A × C
        - P: precipitation (mm → m via /1000)
        - A: roof area (m²)
        - C: runoff coefficient (0.8-0.95 for roofs)
        
        Result in liters: V_L = (P/1000) × A × C × 1000 = P × A × C
        
        Constraints:
        - V_collected ≥ 0 (no negative rain)
        - May be limited by tank available space
        """
        potential = precip_mm * self.A * self.C
        available_space = self.V_max - self.V_current
        collected = min(potential, available_space)
        overflow = max(0, potential - available_space)
        return collected, overflow
    
    def apply_consumption(self, water_consumed_liters):
        """
        Deduct water from tank.
        Check for shortage if tank can't meet demand.
        """
        available = min(self.V_current, water_consumed_liters)
        shortage = max(0, water_consumed_liters - self.V_current)
        self.V_current -= available
        return available, shortage
    
    def daily_update(self, precip_mm, water_demand_liters):
        """
        Single day timestep:
        1. Collect rainfall
        2. Deduct consumption
        3. Update tank state
        4. Track metrics (yield, shortage, overflow)
        """
        pass
```

#### 2.2 Annual Aggregation

```python
class AnnualAggregator:
    """
    Roll up daily results to annual KPIs.
    """
    
    def aggregate(self, daily_results):
        """
        Compute metrics:
        
        WATER YIELD:
        - Total annual collection (m³)
        - Reliability (% of days with sufficient water)
        - Average daily yield (L)
        
        TANK PERFORMANCE:
        - Overflow volume (m³) [wasted water]
        - Average tank level (L)
        - Tank utilization (%)
        - Number of shortage days
        
        EXTREMES:
        - Max 7-day dry period
        - Max daily overflow
        - Shortage frequency (%)
        """
        pass
```

### Stage 3: Model Integration & Validation

```python
class IntegratedModel:
    """
    Combines rainfall + harvesting models with full uncertainty.
    """
    
    def simulate_day(self, day_index, rainfall_series, demand_series):
        """Daily timestep combining all models."""
        pass
    
    def simulate_year(self, rainfall_series, demand_profile):
        """Full 365-day simulation."""
        pass
    
    def monte_carlo(self, n_simulations=1000, scenario='base'):
        """
        Run n_simulations with different:
        1. Random rainfall seeds
        2. Climate scenarios
        3. Building characteristics (sampled uncertainties)
        
        Output: Distribution of outcomes (quantiles, moments, extremes)
        """
        pass
```

---

## 🔌 API INTEGRATION DESIGN

### 1. Meteostat API Client

```python
# backend/services/weather_service.py

class MeteostatWeatherService:
    """
    Production-grade Meteostat client with:
    - Retry logic & exponential backoff
    - Caching layer
    - Rate limiting
    - Error recovery
    """
    
    def __init__(self, cache_service, logger):
        self.session = aiohttp.ClientSession()
        self.cache = cache_service
        self.logger = logger
        self.max_retries = 3
        self.backoff_factor = 2
    
    async def fetch_daily_precipitation(
        self, 
        lat: float, 
        lon: float, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch daily precipitation from Meteostat API.
        
        Workflow:
        1. Check Redis cache (key = f"meteo_{lat}_{lon}_{start}_{end}")
        2. If cache hit: return cached DataFrame
        3. If miss: attempt API call
        4. Implement retry with exponential backoff
        5. Validate response (check completeness, ranges)
        6. Cache result in Redis (TTL = 30 days)
        7. Also archive to InfluxDB (time-series DB)
        8. Return DataFrame
        
        Error handling:
        - 429 (rate limit) → wait & retry
        - 5xx → backoff & retry
        - 4xx non-rate-limit → return cached fallback
        - Timeout (>30s) → use last known good data
        """
        
        cache_key = self._cache_key(lat, lon, start_date, end_date)
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            self.logger.info(f"Cache hit: {cache_key}")
            return pd.read_json(cached)
        
        # API call with retries
        for attempt in range(self.max_retries):
            try:
                data = await self._api_call(lat, lon, start_date, end_date)
                self.logger.info(f"API success on attempt {attempt + 1}")
                
                # Validate
                validated = self._validate_data(data)
                
                # Cache it (30 days TTL)
                await self.cache.set(cache_key, validated.to_json(), ttl=2592000)
                
                return validated
            
            except RateLimitException:
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(f"Rate limited. Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    # Fall back to PostgreSQL historical average
                    return await self._get_historical_average(lat, lon)
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validation checklist:
        ✓ 365 days coverage (or 366 leap)
        ✓ Precipitation in [0, 500] mm
        ✓ No more than 10% missing values
        ✓ Temporal continuity (no date gaps)
        ✓ Reasonable averages (climate-zone appropriate)
        """
        pass
```

### 2. Geospatial API Integration (OSM/Overpass)

```python
class BuildingService:
    """
    Fetch building geometries & characteristics.
    """
    
    async def fetch_buildings_in_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1.0
    ) -> gpd.GeoDataFrame:
        """
        1. Query Overpass API (or local OSM database)
        2. Extract geometries (building footprints)
        3. Calculate roof area = polygon area
        4. Extract tags (height, roof:material, etc.)
        5. Estimate/standardize roof area & height
        6. Cache in PostGIS
        7. Return GeoDataFrame with columns:
           - geometry, roof_area_m2, height_m, osm_id, roof_material
        """
        pass
    
    def calculate_roof_area(self, geometry):
        """Convert OSM geometry → roof area (m²)."""
        # geometry is a shapely.Polygon
        # Use pyproj to get area in m²
        pass
```

### 3. FastAPI Route Examples

```python
# backend/routers/location.py

@router.post("/location/select")
async def select_location(
    request: LocationRequest,  # lat, lon, location_name
    weather_svc: WeatherService = Depends(),
    location_svc: LocationService = Depends()
) -> LocationResponse:
    """
    User clicks a location on map.
    Backend:
    1. Geocode & validate coordinates
    2. Fetch available weather data
    3. Fetch nearby buildings (if OSM enabled)
    4. Return: weather profile + building preview
    """
    pass

@router.post("/simulation/run")
async def run_simulation(
    request: SimulationRequest,  # location, parameters
    sim_svc: SimulationService = Depends()
) -> SimulationResponse:
    """
    Execute full simulation pipeline:
    1. Fetch & prep weather data
    2. Fit rainfall model
    3. Generate synthetic rainfall
    4. Run harvesting simulation (365 days)
    5. Compute KPIs
    6. Return results + visualizations
    """
    pass

@router.get("/simulation/{sim_id}/results")
async def get_simulation_results(sim_id: str) -> ResultsResponse:
    """Retrieve cached results (Redis → GraphQL/JSON)."""
    pass
```

---

## 🔄 SIMULATION ENGINE (ENHANCED)

### Architecture

```python
class SimulationEngine:
    """
    Orchestrates complete simulation pipeline.
    Handles single runs + multi-scenario analysis.
    """
    
    def __init__(self, config: SimulationConfig):
        self.rainfall_model = RainfallModel(config)
        self.harvesting_model = HarvestingModel(config)
        self.analytics = AnalyticsEngine(config)
    
    # ===== SINGLE SIMULATION RUN =====
    
    def run_single(
        self,
        rainfall_series: np.ndarray,
        parameters: SimParameters
    ) -> SimulationResult:
        """
        Execute one deterministic 365-day simulation.
        
        Daily loop (t = 0 to 364):
        1. P_t ← rainfall_series[t]
        2. Collected, Overflow ← harvesting_model.collect(P_t)
        3. Demand_t ← calculate_demand(parameters, t)
        4. Supplied, Shortage ← harvesting_model.consume(Demand_t)
        5. Tank.update(Collected, Supplied, Overflow)
        6. Log state: {V_tank, collected, supplied, shortage, overflow}
        
        Return: SimulationResult with full daily timeseries + KPIs
        """
        pass
    
    # ===== MONTE CARLO UNCERTAINTY QUANTIFICATION =====
    
    def run_monte_carlo(
        self,
        n_simulations: int = 1000,
        parameters: SimParameters,
        seed: int = 42
    ) -> UncertaintyResult:
        """
        Run n_simulations with uncertainty.
        
        For each simulation:
        1. Generate synthetic rainfall (different seed)
        2. Optionally perturb parameters (Gaussian noise)
        3. Execute single run
        4. Collect KPI outcomes
        
        After all runs:
        1. Compute statistics (mean, std, quantiles)
        2. Generate PDF plots
        3. Return: distribution of annual water yield, etc.
        
        Use: multiprocessing or asyncio for parallelization
        """
        pass
    
    # ===== SCENARIO ANALYSIS =====
    
    def run_scenarios(
        self,
        base_parameters: SimParameters,
        scenarios: List[ScenarioConfig]
    ) -> ScenarioResults:
        """
        Compare outcomes across climate/building scenarios.
        
        Example scenarios:
        [
            {"name": "moderate_climate", "precip_change": -0.10, "intensity_change": +0.15},
            {"name": "large_roof", "roof_area": 1000},
            {"name": "small_tank", "tank_capacity": 10000}
        ]
        
        For each scenario:
        1. Modify base_parameters
        2. Run Monte Carlo (100-500 iterations)
        3. Compute expected values & confidence intervals
        
        Output: Comparative analysis (waterfall, heatmap, sensitivity)
        """
        pass
    
    # ===== SENSITIVITY ANALYSIS =====
    
    def sensitivity_analysis(
        self,
        base_parameters: SimParameters,
        parameters_to_vary: List[str],
        variation_range: float = 0.2  # ±20%
    ) -> SensitivityResult:
        """
        Quantify impact of each parameter on outcomes.
        
        For each parameter (e.g., roof_area, tank_capacity, efficiency):
        1. Vary it by ±variation_range
        2. Run simulation (deterministic or MC)
        3. Compute delta KPIs
        4. Rank by impact (one-way sensitivity)
        
        Optional: Interaction effects (two-way sensitivity)
        
        Output: Tornado chart, importance ranking
        """
        pass
```

### Multi-Scenario Execution

```python
class ParallelSimulationRunner:
    """
    Execute multiple scenarios in parallel.
    """
    
    def __init__(self, n_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=n_workers)
    
    def run_batch(self, sim_configs: List[SimConfig]) -> List[SimResult]:
        """
        Distribute simulations across workers.
        Use when: multiple buildings, multiple scenarios, Monte Carlo.
        """
        futures = [
            self.executor.submit(single_simulation, config)
            for config in sim_configs
        ]
        return [f.result() for f in futures]
```

---

## 🎨 UI/UX COMPONENT ARCHITECTURE

### Page Structure (Streamlit Multi-Page)

```
📊 DASHBOARD (Home)
├── 🎯 Key Metrics Cards
│   ├── Annual Water Yield (m³)
│   ├── System Reliability (%)
│   ├── Cost Savings ($/year)
│   └── ROI (years)
├── 🌍 Quick Location Select
│   ├── Map widget (Folium)
│   └── Recent locations
└── 📈 Quick Overview Charts
    ├── Rainfall pattern (annual)
    ├── Tank levels (daily)
    └── Shortage frequency

1️⃣ 🗺️ LOCATION SELECTION
├── 📍 Interactive Map Widget
│   ├── Folium base (2D)
│   ├── Click-to-select
│   ├── Coordinate display
│   └── Search auto-complete (Nominatim)
├── 📋 Location Details Panel
│   ├── Latitude / Longitude
│   ├── Climate zone (auto-detect)
│   ├── Country/Region metadata
│   └── Historical weather summary
└── 🏗️ Building Geometry
    ├── OSM building preview (if available)
    ├── Manual area input
    └── Roof material selector

2️⃣ 📊 DATA EXPLORER
├── 🌧️ Weather Data Viewer
│   ├── Historical precipitation (10+ years)
│   ├── Interactive time-series plot
│   ├── Monthly aggregation
│   ├── Heatmap (daily × year)
│   └── Download CSV
├── 📈 Statistical Analysis
│   ├── Distribution plots (Bernoulli, Gamma fit)
│   ├── Q-Q plots (validation)
│   └── Correlation matrix
└── 🔍 Data Quality Report
    ├── Completeness (%)
    ├── Missing value handling
    └── Outlier detection results

3️⃣ 🔬 MODEL ANALYSIS
├── ⚙️ Rainfall Model Builder
│   ├── Distribution fit visualization
│   ├── Parameter display (p, α, β)
│   ├── Validation metrics (KS test p-value)
│   ├── Seasonal chooser (monthly / annual)
│   └── Preview synthetic series
├── 🎰 Stochastic Simulation
│   ├── Generate synthetic rainfall
│   ├── Compare to historical
│   └── Export synthetic series
└── 🌡️ Climate Scenarios
    ├── Scenario selector (optimistic/moderate/pessimistic)
    ├── Parameter adjustment display
    └── Outcome comparison chart

4️⃣ ⚙️ SCENARIO BUILDER
├── 🏠 Building Parameters
│   ├── Roof area slider (50-5000 m²)
│   ├── Efficiency slider (0.7-0.95)
│   ├── Tank capacity slider (1000-500000 L)
│   ├── Initial tank level (%)
│   └── [Advanced] Building height, orientation
├── 👥 Consumption Profile
│   ├── Daily water demand (fixed or variable)
│   ├── Hourly profile (business hours, etc.)
│   ├── Seasonal adjustment
│   └── Scenario preset buttons
├── 🎯 Simulation Control
│   ├── Run deterministic / Monte Carlo toggle
│   ├── # Simulations slider (for MC)
│   ├── Random seed input
│   └── [RUN SIMULATION] button
└── 📊 Results Panel
    ├── Key metrics (annual yield, reliability, cost)
    ├── Time-series charts (live update)
    ├── Export results (CSV, JSON)
    └── Share/bookmark scenario

5️⃣ 🏢 BUILDING PORTFOLIO
├── 🏗️ Multi-Building Manager
│   ├── Add building (manual or OSM import)
│   ├── Building cards (name, area, current results)
│   ├── Bulk parameter editing
│   └── Compare buildings (side-by-side)
├── 📍 Map View (Pydeck 3D)
│   ├── 3D building extrusions
│   ├── Color by water yield / reliability
│   ├── Click to select & highlight
│   └── Heatmap layer (water potential)
├── 📊 Portfolio Analytics
│   ├── Total annual yield (all buildings)
│   ├── Combined tank optimization
│   ├── Risk analysis (shortage probability across portfolio)
│   └── Comparative ranking
└── 💾 Portfolio Export
    ├── Shapefile/GeoJSON (buildings + results)
    ├── Portfolio summary PDF
    └── Data export (Excel with all metrics)

6️⃣ 💰 ECONOMICS
├── 💵 Cost Analysis
│   ├── Initial investment (tank, materials)
│   ├── Annual maintenance
│   ├── Operating costs (electricity, if pumped)
│   └── Avoided water cost (based on local rates)
├── 📈 Financial Metrics
│   ├── NPV calculator (10-year horizon)
│   ├── IRR (internal rate of return)
│   ├── Payback period
│   ├── Sensitivity to water price
│   └── Break-even roof area
├── 📊 Charts
│   ├── Cash flow timeline
│   ├── Cumulative savings
│   ├── ROI sensitivity (roof area vs tank size)
│   └── Climate scenario impact on ROI
└── 💼 Report Generation
    ├── Business case summary
    ├── Executive summary PDF
    ├── Feasibility recommendation
    └── Share with stakeholders

```

### Component Hierarchy

```python
# frontend/components/map_widget.py
class MapWidget:
    """Reusable geographic selection widget."""
    
    def render(self):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Folium map
            m = folium.Map(
                location=[latitude, longitude],
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            # Add click handler
            folium.LatLngPopup().add_to(m)
            st_folium(m, width=700, height=500)
        
        with col2:
            # Coordinate display
            st.metric("Latitude", f"{lat:.4f}")
            st.metric("Longitude", f"{lon:.4f}")
            # Recent locations
            st.write("**Recent:**")
            for name, coords in recent_locations:
                if st.button(name):
                    st.session_state.location = coords

# frontend/components/3d_visualizer.py
class ThreeDVisualizer:
    """Pydeck 3D visualization of buildings + water."""
    
    def render(self, buildings, simulation_results):
        """
        Create layers:
        1. Building scatterplot (3D extrusions by roof area)
        2. Water yield heatmap
        3. Tank status (color: empty→red, full→blue)
        """
        layers = [
            pdk.Layer(
                'GeoJsonLayer',
                data=buildings_geojson,
                pickable=True,
                extruded=True,
                wireframe=True,
                get_elevation=lambda x: x['properties']['water_yield_kL'] * 10,
                get_fill_color=lambda x: interpolate_color(x['properties']['reliability'])
            ),
            # Tank status layer
            pdk.Layer(
                'ScatterplotLayer',
                data=tanks_df,
                get_position=['lon', 'lat'],
                get_fill_color=lambda x: tank_color(x['fill_percentage']),
                get_radius=500
            )
        ]
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
            initial_view_state=pdk.ViewState(...),
            layers=layers
        ))

# frontend/components/chart_builder.py
class ChartBuilder:
    """Time-series & statistical visualizations."""
    
    def daily_yield_chart(self, daily_results):
        """Stacked bar: rainfall, collected, supplied, shortage."""
        pass
    
    def tank_level_chart(self, tank_levels):
        """Line chart: tank fill over time."""
        pass
    
    def monthly_aggregation(self, daily_results):
        """Monthly totals: yield, shortage, overflow."""
        pass
    
    def reliability_distribution(self, monte_carlo_results):
        """Histogram: distribution of annual reliability across runs."""
        pass
```

---

## 📋 IMPLEMENTATION ROADMAP

### PHASE 1: Foundation & Data (Weeks 1-4)

**Goal**: Production data pipeline + model validation

#### Sprint 1.1: Project Setup & Infrastructure
- [ ] Set up project structure (poetry/pip, Git, CI/CD skeleton)
- [ ] Create Docker dev environment (PostgreSQL, Redis, InfluxDB)
- [ ] Set up logging, error handling middleware
- [ ] Database schema design & Alembic migrations
- [ ] Environment config (.env, settings.yaml)

#### Sprint 1.2: Data Ingestion
- [ ] Build Meteostat API client (with caching, retry logic)
- [ ] Implement data validation & cleaning pipeline
- [ ] Test with multiple locations (5+ cities)
- [ ] Populate PostgreSQL + InfluxDB
- [ ] Build data health dashboard (missing %, outliers, etc.)

#### Sprint 1.3: Model Fitting & Validation
- [ ] Implement rainfall distribution fitting (Bernoulli + Gamma)
- [ ] Goodness-of-fit tests (KS, Anderson-Darling)
- [ ] Seasonal decomposition (monthly parameters)
- [ ] Validation plots (Q-Q, ACF/PACF)
- [ ] Pre-compute model parameters for test locations

**Deliverable**: 
- Data pipeline with ≥95% data quality
- Rainfall models fitted & validated for 10 locations
- Architecture documentation updated

---

### PHASE 2: Simulation Engine (Weeks 5-8)

**Goal**: Complete simulation + analytics

#### Sprint 2.1: Core Simulation
- [ ] Implement harvesting model (collection, tank, consumption)
- [ ] Single-run simulation (365 days deterministic)
- [ ] Daily state tracking & KPI calculation
- [ ] Handle edge cases (empty tank, overflow, leap years)

#### Sprint 2.2: Uncertainty & Scenarios
- [ ] Monte Carlo framework (n_simulations, parallel execution)
- [ ] Climate scenario builder (frequency/intensity adjustments)
- [ ] Sensitivity analysis (tornado charts)
- [ ] Parameter uncertainty (Gaussian perturbations)

#### Sprint 2.3: Analytics Engine
- [ ] Annual aggregation (yield, reliability, costs)
- [ ] Extreme event detection (max dry period, flood risk)
- [ ] Time-series decomposition (trend, seasonality)
- [ ] Comparison analytics (comparing scenarios)

**Deliverable**:
- Complete simulation engine with validation tests
- Scenario analysis for 3+ climate futures
- Performance benchmark (target: 1000 MC runs < 30s)

---

### PHASE 3: API Layer (Weeks 9-12)

**Goal**: Production FastAPI backend

#### Sprint 3.1: FastAPI Setup
- [ ] Route design (location, weather, simulation, buildings, scenarios)
- [ ] Pydantic models for request/response validation
- [ ] Middleware (auth, logging, error handling, rate limiting)
- [ ] OpenAPI/Swagger documentation

#### Sprint 3.2: Service Layer
- [ ] LocationService (geocoding, validation)
- [ ] WeatherService (Meteostat client + caching)
- [ ] SimulationService (orchestration)
- [ ] AnalyticsService (report generation)

#### Sprint 3.3: Caching & Performance
- [ ] Redis caching strategy (weather, model parameters, results)
- [ ] Database query optimization (indexes, connection pooling)
- [ ] API response serialization
- [ ] Load testing (locust or k6)

**Deliverable**:
- FastAPI backend with 20+ endpoints
- API documentation (OpenAPI/Swagger)
- Performance: ≤500ms response time (p95)

---

### PHASE 4: Frontend - Location & Data (Weeks 13-16)

**Goal**: Interactive map + data explorer

#### Sprint 4.1: Map Interface
- [ ] Folium-based 2D map with click-to-select
- [ ] Coordinate geocoding (Nominatim reverse lookup)
- [ ] Location history (sidebar recent locations)
- [ ] Search auto-complete

#### Sprint 4.2: Data Explorer
- [ ] Historical weather visualization (10+ years)
- [ ] Monthly aggregation heatmaps
- [ ] Statistical plots (distributions, Q-Q)
- [ ] Data quality report
- [ ] CSV export

#### Sprint 4.3: Streamlit Pages 1-2
- [ ] Page 1: Home dashboard (KPI cards, recent runs)
- [ ] Page 2: Location selection (fully interactive)

**Deliverable**:
- Pages 1-2 fully functional & styled
- Responsive design (mobile-friendly)
- 0% missing features

---

### PHASE 5: Frontend - Simulation & Scenarios (Weeks 17-20)

**Goal**: Scenario builder + advanced controls

#### Sprint 5.1: Model Analysis Page
- [ ] Rainfall model visualization (fit, distributions)
- [ ] Synthetic series preview
- [ ] Validation metrics display
- [ ] Seasonal selector

#### Sprint 5.2: Scenario Builder
- [ ] Parameter sliders (roof area, efficiency, tank, demand)
- [ ] Preset scenarios (residential, commercial, industrial)
- [ ] Climate scenario selector
- [ ] Run controls (deterministic vs Monte Carlo)

#### Sprint 5.3: Results Visualization
- [ ] Real-time simulation progress
- [ ] KPI dashboard (yield, reliability, cost, ROI)
- [ ] Time-series charts (daily, monthly aggregations)
- [ ] Export (CSV, JSON, PDF)

**Deliverable**:
- Pages 3-4 fully functional
- Live simulation progress indicator
- 500+ scenarios pre-computed & cached

---

### PHASE 6: Frontend - Advanced & Portfolio (Weeks 21-24)

**Goal**: Multi-building + economics

#### Sprint 6.1: Building Portfolio
- [ ] Multi-building manager (CRUD)
- [ ] 3D Pydeck visualization
- [ ] Portfolio-level analytics
- [ ] Batch export (GeoJSON, Shapefile)

#### Sprint 6.2: Economics Module
- [ ] Cost calculator (investment, maintenance, water savings)
- [ ] NPV/IRR/Payback computation
- [ ] Financial sensitivity analysis
- [ ] Business case generator

#### Sprint 6.3: Polish & Testing
- [ ] UI/UX refinement (accessibility, performance)
- [ ] End-to-end testing (Selenium)
- [ ] Performance optimization
- [ ] Documentation

**Deliverable**:
- Pages 5-6 fully functional
- Full product MVP ready for beta
- All 6 pages integrated & tested

---

### PHASE 7: Advanced Features & Deployment (Weeks 25-28)

**Goal**: Optimization + cloud deployment

#### Sprint 7.1: Advanced Modeling
- [ ] Tank optimization (capacity sizing for reliability threshold)
- [ ] Multi-building optimization (shared tank scenarios)
- [ ] Climate adaptation strategies (roof area, tank size)
- [ ] Economic optimization (cost-benefit trade-offs)

#### Sprint 7.2: Geospatial Enhancements
- [ ] OSM building integration (auto-fetch from Overpass)
- [ ] DEM-based height estimation
- [ ] Administrative boundary analysis (city/region scale)
- [ ] Country-scale heatmap generation

#### Sprint 7.3: Deployment & DevOps
- [ ] Docker containerization (backend, frontend)
- [ ] Kubernetes deployment specs
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring & alerting (Prometheus, Grafana)
- [ ] Database backup strategy

**Deliverable**:
- Kubernetes-ready deployment
- CI/CD fully automated
- Monitoring dashboard live

---

### PHASE 8: Production Hardening (Weeks 29-32)

**Goal**: Enterprise-ready system

#### Sprint 8.1: Quality Assurance
- [ ] Unit tests (target: 85%+ coverage)
- [ ] Integration tests (pipeline end-to-end)
- [ ] Performance tests (SLA compliance)
- [ ] Security audit (OWASP top 10)

#### Sprint 8.2: User Acceptance Testing
- [ ] Beta testing with 10-20 users
- [ ] Feedback incorporation
- [ ] Edge case handling
- [ ] Documentation updates

#### Sprint 8.3: Launch Preparation
- [ ] Knowledge transfer documentation
- [ ] User onboarding guides
- [ ] Support infrastructure setup
- [ ] Analytics/telemetry
- [ ] Public launch

**Deliverable**:
- Production-grade system
- Release notes & changelog
- Ready for public/enterprise use

---

## 📈 SCALABILITY CONSIDERATIONS

### Vertical Scaling (Single Machine)

```
Bottleneck        | Solution                    | Implementation
──────────────────┼─────────────────────────────┼──────────────────────
Memory (100s GB)  | Stream processing           | Dask, Apache Spark
CPU (1000s cores)| GPU computation             | CuPy, RAPIDS
Disk IOPS        | SSD + NVMe                  | AWS EBS GP3/io2
Database queries | Query optimization          | Indexes, materialization
```

### Horizontal Scaling (Distributed)

```
Layer              | Scaling Strategy           | Technology
──────────────────┼───────────────────────────┼──────────────────────
Frontend           | CDN + Auto-scaling groups  | CloudFlare, ALB + ASG
API                | Load balancer + replicas  | Kubernetes HPA
Simulation         | Distributed task queue    | Celery + Redis, Dask
Cache              | Multi-tier                | Redis Cluster, Memcached
Database (OLTP)    | Replication + sharding    | PostgreSQL Replication
Database (OLAP)    | Star schema               | Data warehouse (BigQuery)
Time-series        | Sharding by station       | InfluxDB Cluster
Geospatial         | PostGIS clustering        | Partitioned indexes
```

### Load Estimation (Year 1)

```
Scenario: 1,000 locations × 100 buildings = 100,000 simulations/month

Monthly Volume:
- Weather API calls: 1,000 (one per location, cached)
- Simulation runs: 100,000 (1355/day)
- Results stored: 100,000 × 365 days × 24 hours = 876M data points
- Storage: ~40 GB (time-series DB)

Cost Breakdown (AWS):
- RDS (t3.large): $250/month
- InfluxDB (managed): $500/month
- ECS Fargate (2vCPU, 4GB): $150/month
- S3 + CloudFront: $100/month
- ────────────────────────
- Total: ~$1,000/month
```

### Caching Strategy

```
Layer              | TTL       | Key Pattern               | Use Case
──────────────────┼───────────┼──────────────────────────┼──────────────────
Weather Data      | 30 days   | meteo_{lat}_{lon}_{date} | API response
Model Parameters  | 90 days   | model_{lat}_{lon}_{month}| Rainfall params
Simulation Result | 1 day     | sim_{building_id}_{date} | User dashboard
API Response      | 1 hour    | api_{endpoint}_{hash}    | Common queries
Session State     | 24 hours  | session_{user_id}        | User session
```

---

## ⚠️ RISK ANALYSIS & MITIGATION

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Meteostat API unavailability** | High | Medium | Implement fallback to OpenWeatherMap & NOAA; cache 10-year history |
| **Weather data gaps (missing months)** | High | Medium | Multiple imputation algorithm; flag anomalies in UI |
| **Simulation computation time (>1min)** | Medium | Medium | Implement memoization + pre-computed results; GPU acceleration |
| **Geospatial database performance** | Medium | Low | Spatial indexing (GiST); partition by region |
| **Model overfitting to historical data** | Medium | Low | Cross-validation; out-of-sample testing; climate validation |
| **Concurrency/race conditions** | Medium | Low | Async/await + connection pooling; pessimistic locking for critical sections |

### Data Quality Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Missing precipitation data** | High | High | Linear interpolation (gaps <3d), seasonal avg (3-30d), multiple imputation (>30d) |
| **Unrealistic parameter values (roof area >10,000m²)** | Medium | Medium | Input validation; semantic checks (area vs building footprint) |
| **Inconsistent geographic data** | Medium | Low | Coordinate validation (EPSG:4326); geometry simplification |
| **Climate scenarios unrealistic** | Low | Medium | Peer-review with climatologists; compare vs CMIP6/IPCC |

### Business/Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **User misinterprets results** | High | Medium | Clear documentation; confidence intervals & uncertainty bands |
| **API rate limiting (Meteostat)** | Medium | High | Implement retry + exponential backoff; negotiate tier upgrade |
| **Server downtime during critical analysis** | High | Low | Redundancy (multi-AZ); automated failover; SLA monitoring |
| **Data privacy (location, consumption patterns)** | High | Low | Anonymization; encryption at rest/transit; GDPR compliance |

### Model Validation Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Rainfall model underestimating extremes** | High | Medium | Explicitly model tail behavior (GEV); Extreme Value Theory |
| **Harvesting calc diverges from real-world** | Medium | Medium | Field validation with pilot projects; continuous calibration |
| **Tank optimization overly optimistic** | Medium | Low | Monte Carlo with worst-case scenarios; safety factor (1.2×) |
| **Economic assumptions wrong** | Medium | Low | Local market research; parametric sensitivity; payback curve |

### Mitigation Checklist

```python
RISK RESPONSES:
✓ Implement comprehensive logging (ELK stack)
✓ Set up monitoring & alerting (Prometheus + PagerDuty)
✓ Version control for all config/data/code
✓ Automated backups (daily to S3)
✓ Disaster recovery plan (RTO ≤ 1 day)
✓ Documentation of assumptions & limitations
✓ Regular model validation (quarterly)
✓ User feedback loop + A/B testing
✓ Security audit (penetration testing)
✓ Legal review (data handling, liability)
```

---

## 📊 SUCCESS METRICS

### Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response Time (p95) | ≤500ms | CloudWatch / Prometheus |
| Model Validation (KS test p-value) | >0.05 | Quarterly audit |
| Data Completeness | ≥95% | SQL query on timeseries DB |
| Simulation Runtime (365 days, 1000 MC runs) | ≤30s | Benchmark script |
| Cache Hit Ratio | ≥80% | Redis INFO |
| Uptime | ≥99.5% | Monitoring dashboard |

### User KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| User Retention (30 days) | ≥70% | GA / Custom analytics |
| Task Completion Rate | ≥80% | Event tracking |
| Feature Adoption (portfolio view) | ≥50% | Usage analytics |
| User Satisfaction (NPS) | ≥40 | Quarterly survey |
| Support Ticket Volume | ≤5/1000 users | Ticketing system |

### Business KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cost per Simulation | <$0.10 | Cloud billing + logs |
| Model Accuracy (vs real-world) | ±15% | Field validation |
| Time to Production | ≤8 months | Gantt chart |
| Regulatory Compliance | 100% | Audit check |

---

## 📚 DOCUMENTATION ROADMAP

Essential documents to create in parallel with development:

```
docs/
├── ARCHITECTURE.md (THIS FILE)
├── API_REFERENCE.md (OpenAPI auto-generated)
├── DATA_DICTIONARY.md (all fields defined)
├── DEPLOYMENT_GUIDE.md (step-by-step)
├── TROUBLESHOOTING.md (common issues)
├── USER_GUIDE.md (end-user manual)
├── DEVELOPER_GUIDE.md (setup + contribution)
├── OPERATIONS.md (monitoring, backups, scaling)
├── SECURITY.md (data privacy, compliance)
└── CHANGELOG.md (version history)
```

---

## 🎯 NEXT STEPS

1. **Review & Feedback Loop** (1 week)
   - This architecture with stakeholders
   - Adjust priorities based on feedback
   - Lock in requirements

2. **Project Initialization** (1 week)
   - Set up Git repo + CI/CD skeleton
   - Create issue tracker (GitHub/Jira)
   - Assign teams to sprints

3. **Phase 1 Kickoff** (Week 1)
   - Begin project setup + data ingestion
   - Parallel: API design review
   - Parallel: UI/UX mockups (Figma)

---

## 📝 APPENDIX: QUICK REFERENCE

### Key Formulas

```
RAINFALL COLLECTION:
V_collected = P × A × C
- P: precipitation (mm)
- A: roof area (m²)
- C: runoff coefficient (0.7-0.95)
- Result: liters

ANNUAL RELIABILITY:
R = (days_with_sufficient_water) / 365 × 100%

COST-BENEFIT:
NPV = Σ(CF_t / (1+r)^t) - Initial_Cost
- CF_t: annual cash flow (year t)
- r: discount rate (6-10%)

TANK OPTIMIZATION:
Find V_tank that maximizes NPV subject to:
- Initial cost ≤ budget
- Overflow losses ≤ threshold
- Shortage probability ≥ target_reliability
```

### Technology Stack (Final Recommendation)

```
Frontend:     Streamlit + Plotly + Folium + Pydeck
Backend:      FastAPI + Uvicorn + SQLAlchemy
Databases:    PostgreSQL + PostGIS + InfluxDB + Redis
Data Pipeline: Pandas + NumPy + SciPy
Modeling:     Scikit-learn + Statsmodels
Deployment:   Docker + Kubernetes + AWS
Monitoring:   Prometheus + Grafana + CloudWatch
Testing:      Pytest + Locust + Selenium
```

---

**END OF ARCHITECTURE DOCUMENT**

---

*This architecture is designed for**production-grade deployment**. Implementation requires careful attention to data quality, model validation, and user experience. Estimated effort: 8 months (4 teams of 2 engineers each).*

