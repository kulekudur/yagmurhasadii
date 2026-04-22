# 📡 API SPECIFICATION & DATABASE SCHEMA

**Version**: 2.0  
**Last Updated**: April 2026  
**Base URL**: `/api/v1`

---

## 📚 TABLE OF CONTENTS

1. [API Endpoints](#api-endpoints)
2. [Data Models (Pydantic)](#data-models)
3. [Database Schema (ER Diagram)](#database-schema)
4. [Error Responses](#error-responses)
5. [Authentication & Authorization](#auth)
6. [Rate Limiting](#rate-limiting)

---

## 📡 API ENDPOINTS

### Location Management

#### `POST /location/select`
**Purpose**: User selects a location from map  
**Auth**: Optional (anonymous allowed)

**Request**:
```json
{
  "latitude": 41.0082,
  "longitude": 28.9784,
  "location_name": "Istanbul",
  "country": "Turkey"
}
```

**Response (200)**:
```json
{
  "id": 1,
  "latitude": 41.0082,
  "longitude": 28.9784,
  "location_name": "Istanbul",
  "country": "Turkey",
  "climate_zone": "Mediterranean",
  "weather_data_years": 10,
  "created_at": "2026-04-19T10:30:00Z",
  "weather_summary": {
    "annual_precip_mm": 672.5,
    "rainy_days_per_year": 112,
    "max_daily_precip_mm": 145.2,
    "avg_temp_c": 14.8
  }
}
```

**Errors**:
- `400`: Invalid coordinates (outside ±90°, ±180°)
- `503`: Weather data unavailable for location

---

#### `GET /location/{location_id}`
**Purpose**: Retrieve location details with weather profile

**Response (200)**:
```json
{
  "id": 1,
  "latitude": 41.0082,
  "longitude": 28.9784,
  "location_name": "Istanbul",
  "country": "Turkey",
  "weather_profile": {
    "p_rain_annual": 0.307,
    "p_rain_monthly": [0.25, 0.22, 0.28, ...],
    "gamma_shape": 1.98,
    "gamma_scale": 5.12,
    "validation_ks_pvalue": 0.156
  },
  "buildings_count": 5,
  "recent_simulations": [...]
}
```

---

#### `GET /location/search`
**Purpose**: Search locations by name/country

**Query Params**:
- `q`: String (location name or country)
- `limit`: Int (max results, default: 10)

**Response (200)**:
```json
{
  "results": [
    {
      "id": 1,
      "location_name": "Istanbul",
      "country": "Turkey",
      "latitude": 41.0082,
      "longitude": 28.9784,
      "times_used": 5
    },
    ...
  ],
  "total": 42
}
```

---

### Weather & Data Integration

#### `GET /weather/{location_id}/daily`
**Purpose**: Fetch daily precipitation data (10+ years)

**Query Params**:
- `start_date`: YYYY-MM-DD (default: 10 years ago)
- `end_date`: YYYY-MM-DD (default: today)
- `format`: csv|json (default: json)

**Response (200)**:
```json
{
  "location_id": 1,
  "start_date": "2014-01-01",
  "end_date": "2024-04-19",
  "total_days": 3652,
  "completeness_pct": 97.3,
  "data": [
    {"date": "2014-01-01", "precip_mm": 0.0, "temp_c": 8.5, "humidity_pct": 62},
    {"date": "2014-01-02", "precip_mm": 12.4, "temp_c": 7.2, "humidity_pct": 75},
    ...
  ],
  "statistics": {
    "mean_precip": 1.84,
    "median_precip": 0.0,
    "std_precip": 7.32,
    "min_precip": 0.0,
    "max_precip": 145.2,
    "rainy_days": 1125
  }
}
```

**Errors**:
- `404`: Location not found
- `503`: Data provider unavailable (with cache fallback)

---

#### `GET /weather/{location_id}/monthly`
**Purpose**: Aggregated monthly precipitation

**Response (200)**:
```json
{
  "location_id": 1,
  "data": [
    {
      "year": 2014,
      "month": 1,
      "total_precip_mm": 45.6,
      "rainy_days": 8,
      "max_daily": 22.3
    },
    ...
  ]
}
```

---

#### `POST /weather/{location_id}/validate`
**Purpose**: Generate data quality report

**Request**: (empty body)

**Response (200)**:
```json
{
  "location_id": 1,
  "completeness": {
    "total_days": 3652,
    "days_with_data": 3548,
    "percent": 97.1
  },
  "missing_periods": [
    {
      "start_date": "2015-03-05",
      "end_date": "2015-03-07",
      "duration_days": 3,
      "method_filled": "linear_interpolation"
    }
  ],
  "outliers_detected": 12,
  "quality_score": 9.2,
  "recommendation": "Data quality is excellent. Proceed with modeling."
}
```

---

### Simulation & Modeling

#### `POST /simulation/run`
**Purpose**: Execute single simulation (365 days)

**Request**:
```json
{
  "location_id": 1,
  "building": {
    "roof_area_m2": 500,
    "efficiency": 0.85,
    "height_m": 15,
    "roof_material": "concrete"
  },
  "tank": {
    "capacity_liters": 50000,
    "initial_level_pct": 50,
    "overflow_loss_pct": 5
  },
  "consumption": {
    "daily_demand_liters": 1000,
    "hourly_profile": [0, 0, 0, 50, 100, 200, 300, 250, 150, 100, 100, 100, 100, 80, 60, 40, 20, 10, 5, 0, 0, 0, 0, 0],
    "seasonal_adjustment": [1.0, 0.95, 0.9, 0.8, 0.7, 0.8, 1.0, 1.1, 1.0, 0.95, 0.95, 1.0]
  },
  "simulation": {
    "type": "deterministic",
    "rainfall_seed": 42,
    "scenario": "base"
  }
}
```

**Response (202 - Accepted)**:
```json
{
  "simulation_id": "sim_abc123def456",
  "status": "queued",
  "location": {"id": 1, "name": "Istanbul"},
  "created_at": "2026-04-19T10:35:00Z",
  "estimated_completion_seconds": 30,
  "result_url": "/api/v1/simulation/sim_abc123def456/result"
}
```

**Poll Result** (GET `/simulation/{simulation_id}/result`):
```json
{
  "simulation_id": "sim_abc123def456",
  "status": "completed",
  "location_id": 1,
  "run_time_seconds": 2.34,
  "results": {
    "annual_precipitation_mm": 672.5,
    "annual_collected_m3": 285.6,
    "annual_consumed_m3": 365.0,
    "annual_shortage_m3": 79.4,
    "annual_overflow_m3": 22.1,
    "reliability_pct": 78.3,
    "avg_tank_level_pct": 42.5,
    "max_dry_period_days": 18,
    "shortages_count": 23
  },
  "daily_series": [
    {
      "day_of_year": 1,
      "precip_mm": 0.0,
      "collected_L": 0,
      "consumed_L": 950,
      "shortage_L": 950,
      "overflow_L": 0,
      "tank_level_L": 0
    },
    ...
  ],
  "monthly_aggregation": [
    {
      "month": 1,
      "precip_mm": 45.6,
      "collected_m3": 19.2,
      "consumed_m3": 29.5,
      "shortage_m3": 10.3,
      "avg_tank_level_L": 15000
    },
    ...
  ]
}
```

**Errors**:
- `400`: Invalid parameters (roof area outside range, etc.)
- `404`: Location not found
- `429`: Rate limit exceeded
- `503`: Simulation queue full

---

#### `POST /simulation/monte-carlo`
**Purpose**: Run n_simulations with uncertainty quantification

**Request**:
```json
{
  "location_id": 1,
  "building": { ... },  // Same as /run
  "tank": { ... },
  "consumption": { ... },
  "monte_carlo": {
    "n_simulations": 500,
    "parameter_uncertainty": {
      "roof_area_variation": 0.05,
      "efficiency_variation": 0.03,
      "tank_capacity_variation": 0.10
    },
    "scenarios": ["base", "moderate_climate", "pessimistic_climate"],
    "seed": 42
  }
}
```

**Response (202 - Accepted)**:
```json
{
  "simulation_id": "mc_xyz789abc123",
  "status": "queued",
  "simulation_count": 1500,  // 500 × 3 scenarios
  "estimated_completion_seconds": 180,
  "progress_url": "/api/v1/simulation/mc_xyz789abc123/progress"
}
```

**Poll Progress** (GET `/simulation/{simulation_id}/progress`):
```json
{
  "simulation_id": "mc_xyz789abc123",
  "status": "running",
  "completed_simulations": 425,
  "total_simulations": 1500,
  "percent_complete": 28.3,
  "elapsed_seconds": 45,
  "estimated_remaining_seconds": 110,
  "current_scenario": "moderate_climate"
}
```

**Final Result** (GET `/simulation/{simulation_id}/result`):
```json
{
  "simulation_id": "mc_xyz789abc123",
  "status": "completed",
  "scenarios": {
    "base": {
      "annual_yield_m3": {
        "mean": 285.6,
        "std": 18.5,
        "p5": 251.2,
        "p25": 274.3,
        "median": 286.1,
        "p75": 297.8,
        "p95": 321.5
      },
      "reliability_pct": {
        "mean": 78.3,
        "std": 4.2,
        "p5": 68.1,
        "p95": 85.2
      },
      "shortage_m3": {...},
      "overflow_m3": {...}
    },
    "moderate_climate": {
      "annual_yield_m3": {
        "mean": 272.1,
        "std": 20.3,
        "p5": 232.4,
        "p95": 308.9
      },
      ...
    },
    "pessimistic_climate": {...}
  },
  "comparison": {
    "base_vs_moderate": {
      "yield_change_pct": -4.7,
      "reliability_change_pct": -6.2
    },
    "moderate_vs_pessimistic": {
      "yield_change_pct": -8.3,
      "reliability_change_pct": -9.1
    }
  }
}
```

---

#### `POST /simulation/sensitivity`
**Purpose**: Parameter sensitivity analysis

**Request**:
```json
{
  "location_id": 1,
  "building": { ... },
  "tank": { ... },
  "consumption": { ... },
  "sensitivity": {
    "parameters": ["roof_area", "efficiency", "tank_capacity"],
    "variation_range": 0.20  // ±20%
  }
}
```

**Response (200)**:
```json
{
  "sensitivity_results": {
    "annual_yield": {
      "roof_area": {
        "elasticity": 1.0,  // 1% change in area → 1% change in yield
        "rank": 1
      },
      "efficiency": {
        "elasticity": 1.0,
        "rank": 2
      },
      "tank_capacity": {
        "elasticity": 0.18,  // Small effect (tank rarely filled)
        "rank": 3
      }
    },
    "reliability": {
      "tank_capacity": {
        "elasticity": 0.65,
        "rank": 1
      },
      "roof_area": {
        "elasticity": 0.42,
        "rank": 2
      },
      ...
    }
  },
  "tornado_chart": {
    "base_yield": 285.6,
    "ranges": [
      {"param": "roof_area", "min": 228.5, "max": 342.7},
      {"param": "efficiency", "min": 242.8, "max": 328.4},
      ...
    ]
  }
}
```

---

### Buildings & Portfolio

#### `POST /buildings`
**Purpose**: Create a new building entry

**Request**:
```json
{
  "location_id": 1,
  "name": "Office Block A",
  "roof_area_m2": 800,
  "efficiency": 0.85,
  "height_m": 20,
  "roof_material": "concrete",
  "coordinates": {
    "latitude": 41.0082,
    "longitude": 28.9784
  },
  "tags": ["commercial", "priority"]
}
```

**Response (201)**:
```json
{
  "id": 12,
  "location_id": 1,
  "name": "Office Block A",
  "roof_area_m2": 800,
  "efficiency": 0.85,
  "height_m": 20,
  "coordinates": {
    "latitude": 41.0082,
    "longitude": 28.9784,
    "geometry_wkt": "POLYGON((...))}"
  },
  "created_at": "2026-04-19T10:40:00Z",
  "last_simulation": null
}
```

---

#### `GET /buildings/{building_id}`
**Purpose**: Retrieve building details + latest simulation results

**Response (200)**:
```json
{
  "id": 12,
  "location_id": 1,
  "name": "Office Block A",
  "roof_area_m2": 800,
  "efficiency": 0.85,
  "height_m": 20,
  "created_at": "2026-04-19T10:40:00Z",
  "latest_simulation": {
    "simulation_id": "sim_abc123def456",
    "run_date": "2026-04-19T10:45:00Z",
    "annual_yield_m3": 340.2,
    "reliability_pct": 81.5,
    "shortage_days": 19
  },
  "simulations_count": 12
}
```

---

#### `GET /buildings?location_id=1`
**Purpose**: List all buildings in a location

**Query Params**:
- `location_id`: Required
- `skip`: Offset (default: 0)
- `limit`: Row limit (default: 50)

**Response (200)**:
```json
{
  "total": 5,
  "buildings": [
    {
      "id": 12,
      "name": "Office Block A",
      "roof_area_m2": 800,
      "latest_yield_m3": 340.2,
      "latest_reliability_pct": 81.5
    },
    ...
  ]
}
```

---

#### `PUT /buildings/{building_id}`
**Purpose**: Update building parameters

**Request**:
```json
{
  "roof_area_m2": 850,
  "efficiency": 0.88,
  "height_m": 22
}
```

**Response (200)**: Updated building object

---

#### `DELETE /buildings/{building_id}`
**Purpose**: Delete a building

**Response (204)**: No content

---

#### `POST /portfolios`
**Purpose**: Create a portfolio (collection of buildings)

**Request**:
```json
{
  "name": "Downtown Zone",
  "description": "All buildings in downtown area",
  "building_ids": [12, 13, 14, 15],
  "location_id": 1
}
```

**Response (201)**:
```json
{
  "id": 3,
  "name": "Downtown Zone",
  "description": "All buildings in downtown area",
  "building_ids": [12, 13, 14, 15],
  "location_id": 1,
  "created_at": "2026-04-19T11:00:00Z"
}
```

---

#### `GET /portfolios/{portfolio_id}/analytics`
**Purpose**: Portfolio-level analytics (aggregated)

**Response (200)**:
```json
{
  "portfolio_id": 3,
  "portfolio_name": "Downtown Zone",
  "building_count": 4,
  "summary": {
    "total_roof_area_m2": 3200,
    "combined_annual_yield_m3": 1342.5,
    "combined_annual_shortage_m3": 287.3,
    "portfolio_reliability_pct": 79.8,
    "portfolio_efficiency": 0.86
  },
  "buildings_analysis": [
    {
      "building_id": 12,
      "name": "Office Block A",
      "roof_area_m2": 800,
      "yield_m3": 340.2,
      "reliability_pct": 81.5,
      "yield_ranking": 1
    },
    ...
  ],
  "variance_analysis": {
    "highest_yield": {"building_id": 12, "value": 340.2},
    "lowest_yield": {"building_id": 15, "value": 287.1},
    "coefficient_of_variation": 0.08
  }
}
```

---

### Economics & ROI

#### `POST /economics/roi-analysis`
**Purpose**: Financial viability analysis

**Request**:
```json
{
  "location_id": 1,
  "building": {
    "roof_area_m2": 500,
    "efficiency": 0.85
  },
  "tank": {
    "capacity_liters": 50000
  },
  "economics": {
    "tank_cost": 5000,
    "installation_cost": 2000,
    "maintenance_annual": 500,
    "water_price_per_liter": 0.005,
    "discount_rate": 0.08,
    "analysis_horizon_years": 10
  }
}
```

**Response (200)**:
```json
{
  "annual_water_yield_m3": 285.6,
  "annual_water_yield_liters": 285600,
  "annual_water_cost_avoided": 1428,
  "annual_maintenance_cost": 500,
  "annual_net_benefit": 928,
  "cashflow": [
    {
      "year": 0,
      "cashflow": -7000,
      "cumulative": -7000
    },
    {
      "year": 1,
      "cashflow": 928,
      "cumulative": -6072
    },
    {
      "year": 2,
      "cashflow": 928,
      "cumulative": -5144
    },
    ...
  ],
  "npv_10y": 1234.56,
  "irr": 0.085,
  "payback_period_years": 7.54,
  "roi_percentage": 8.5,
  "recommendation": "Economically viable with moderate returns"
}
```

---

#### `GET /economics/water-price-sensitivity`
**Purpose**: How ROI varies with local water price

**Query Params**:
- `building_id`: Required
- `price_min`: Min price/L (default: 0.002)
- `price_max`: Max price/L (default: 0.020)
- `steps`: # of data points (default: 10)

**Response (200)**:
```json
{
  "sensitivity_curve": [
    {"water_price": 0.002, "payback_years": 18.5, "npv_10y": -2145},
    {"water_price": 0.004, "payback_years": 12.3, "npv_10y": -456},
    {"water_price": 0.006, "payback_years": 9.1, "npv_10y": 1234},
    {"water_price": 0.008, "payback_years": 7.4, "npv_10y": 2923},
    ...
  ],
  "break_even_water_price": 0.0058,
  "current_market_price": 0.005,
  "margin_to_breakeven_pct": -13.6
}
```

---

### Scenarios Management

#### `POST /scenarios`
**Purpose**: Create a climate/parameter scenario

**Request**:
```json
{
  "name": "Pessimistic Climate 2050",
  "description": "IPCC SSP5-8.5 projection for 2050",
  "scenario_type": "climate",
  "parameters": {
    "precip_frequency_change_pct": -15,
    "precip_intensity_change_pct": 30,
    "temperature_change_c": 2.5
  }
}
```

**Response (201)**:
```json
{
  "id": 5,
  "name": "Pessimistic Climate 2050",
  "scenario_type": "climate",
  "parameters": {...},
  "created_at": "2026-04-19T11:30:00Z"
}
```

---

#### `GET /scenarios`
**Purpose**: List all predefined scenarios

**Response (200)**:
```json
{
  "total": 8,
  "scenarios": [
    {
      "id": 1,
      "name": "Base Case",
      "scenario_type": "base",
      "description": "Historical climate averages"
    },
    {
      "id": 2,
      "name": "Optimistic 2050",
      "scenario_type": "climate",
      "description": "IPCC SSP1-1.9 (Paris targets)"
    },
    {
      "id": 3,
      "name": "Moderate 2050",
      "scenario_type": "climate",
      "description": "IPCC SSP2-4.5"
    },
    {
      "id": 4,
      "name": "Pessimistic 2050",
      "scenario_type": "climate",
      "description": "IPCC SSP5-8.5"
    },
    {
      "id": 5,
      "name": "Extreme Events",
      "scenario_type": "extreme",
      "description": "30% increase in extreme rainfall"
    },
    ...
  ]
}
```

---

## 📋 DATA MODELS (Pydantic)

### Core Models

```python
# backend/models/location.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    location_name: str = Field(..., min_length=1, max_length=255)
    country: Optional[str] = None

class LocationResponse(BaseModel):
    id: int
    latitude: float
    longitude: float
    location_name: str
    country: Optional[str]
    climate_zone: Optional[str]
    weather_data_years: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# backend/models/building.py
class BuildingCreate(BaseModel):
    location_id: int
    name: str = Field(..., min_length=1, max_length=255)
    roof_area_m2: float = Field(..., gt=0, le=100000)
    efficiency: float = Field(..., ge=0.5, le=1.0)
    height_m: Optional[float] = Field(None, gt=0, le=500)
    roof_material: Optional[str] = None
    tags: Optional[list[str]] = []

class BuildingResponse(BuildingCreate):
    id: int
    created_at: datetime
    last_simulation_id: Optional[int] = None

# backend/models/simulation.py
class ConsumptionProfile(BaseModel):
    daily_demand_liters: float = Field(..., gt=0)
    hourly_profile: list[float] = Field(default_factory=lambda: [0]*24)
    seasonal_adjustment: list[float] = Field(default_factory=lambda: [1.0]*12)

class SimulationRequest(BaseModel):
    location_id: int
    building: BuildingCreate
    tank: TankSpecification
    consumption: ConsumptionProfile
    simulation: SimulationParams

class SimulationResult(BaseModel):
    simulation_id: str
    status: str  # "completed", "running", "failed"
    results: dict
    daily_series: list[dict]
    monthly_aggregation: list[dict]
    run_time_seconds: float
```

---

## 🗄️ DATABASE SCHEMA (ER Model)

### Entities

```
USERS
├── id (PK)
├── email (UNIQUE)
├── password_hash
├── created_at
├── updated_at

LOCATIONS
├── id (PK)
├── latitude
├── longitude
├── name
├── country
├── climate_zone
├── created_at
├── geometry (PostGIS Point)

BUILDINGS
├── id (PK)
├── location_id (FK → LOCATIONS)
├── user_id (FK → USERS)
├── name
├── roof_area_m2
├── efficiency
├── height_m
├── roof_material
├── geometry (PostGIS Polygon)
├── created_at
├── updated_at

BUILDING_PORTFOLIOS
├── id (PK)
├── user_id (FK → USERS)
├── name
├── description
├── created_at
├── updated_at

PORTFOLIO_BUILDINGS (Join Table)
├── portfolio_id (FK → BUILDING_PORTFOLIOS)
├── building_id (FK → BUILDINGS)

WEATHER_PROFILES
├── id (PK)
├── location_id (FK → LOCATIONS)
├── p_rain_annual
├── p_rain_monthly (JSON array)
├── gamma_shape
├── gamma_scale
├── validation_ks_pvalue
├── fitted_on_date
├── updated_at

DAILY_PRECIPITATION (Time-Series)
├── id (PK)
├── location_id (FK → LOCATIONS)
├── date (DATE)
├── precip_mm
├── temp_c
├── humidity_pct
├── wind_kmh
├── data_quality_flag
├── created_at
├── UNIQUE(location_id, date)

SIMULATION_CONFIGS
├── id (PK)
├── building_id (FK → BUILDINGS)
├── roof_area_m2
├── efficiency
├── tank_capacity_liters
├── tank_initial_level_pct
├── daily_demand_liters
├── created_at

SIMULATION_RESULTS
├── id (PK)
├── config_id (FK → SIMULATION_CONFIGS)
├── location_id (FK → LOCATIONS)
├── simulation_id (UNIQUE)
├── status
├── annual_collection_m3
├── annual_consumption_m3
├── annual_shortage_m3
├── annual_overflow_m3
├── reliability_pct
├── run_time_seconds
├── created_at

DAILY_SIMULATION_STATE (Time-Series)
├── id (PK)
├── result_id (FK → SIMULATION_RESULTS)
├── day_of_year
├── precip_mm
├── collected_L
├── consumed_L
├── shortage_L
├── overflow_L
├── tank_level_L

MONTHLY_AGGREGATION
├── id (PK)
├── result_id (FK → SIMULATION_RESULTS)
├── month_of_year
├── total_precip_mm
├── total_collected_m3
├── total_consumed_m3
├── total_shortage_m3
├── avg_tank_level_L

SCENARIOS
├── id (PK)
├── name
├── description
├── scenario_type
├── parameters (JSON)
├── created_at

SCENARIO_RESULTS (Join Table)
├── id (PK)
├── scenario_id (FK → SCENARIOS)
├── result_id (FK → SIMULATION_RESULTS)
├── outcome_json
```

### SQL Schema (Core Tables)

```sql
-- Locations with geometry
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    latitude DECIMAL(10,8) NOT NULL,
    longitude DECIMAL(11,8) NOT NULL,
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    climate_zone VARCHAR(50),
    geometry GEOMETRY(Point, 4326),  -- PostGIS
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(latitude, longitude)
);

CREATE INDEX idx_locations_geometry ON locations USING GIST(geometry);
CREATE INDEX idx_locations_country ON locations(country);

-- Buildings
CREATE TABLE buildings (
    id SERIAL PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    roof_area_m2 DECIMAL(10,2) NOT NULL,
    efficiency DECIMAL(3,2) NOT NULL CHECK (efficiency >= 0.5 AND efficiency <= 1.0),
    height_m DECIMAL(8,2),
    roof_material VARCHAR(100),
    geometry GEOMETRY(Polygon, 4326),  -- Building footprint
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_buildings_location ON buildings(location_id);
CREATE INDEX idx_buildings_geometry ON buildings USING GIST(geometry);

-- Daily Precipitation (Time-Series)
CREATE TABLE daily_precipitation (
    id BIGSERIAL PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    precip_mm DECIMAL(6,1),
    temp_c DECIMAL(5,2),
    humidity_pct DECIMAL(5,2),
    wind_kmh DECIMAL(5,2),
    data_quality_flag VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(location_id, date)
);

CREATE INDEX idx_daily_precip_location_date ON daily_precipitation(location_id, date);

-- Simulation Results
CREATE TABLE simulation_results (
    id BIGSERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES simulation_configs(id),
    location_id INTEGER NOT NULL REFERENCES locations(id),
    simulation_id VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    annual_collection_m3 DECIMAL(10,3),
    annual_consumption_m3 DECIMAL(10,3),
    annual_shortage_m3 DECIMAL(10,3),
    reliability_pct DECIMAL(5,2),
    run_time_seconds DECIMAL(8,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_simulation_results_created ON simulation_results(created_at DESC);
```

---

## ❌ ERROR RESPONSES

### Standard Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more validation errors occurred",
    "details": [
      {
        "field": "roof_area_m2",
        "message": "Must be between 10 and 10000",
        "constraint": "range"
      }
    ],
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `VALIDATION_ERROR` | 400 | Invalid input parameters |
| `LOCATION_NOT_FOUND` | 404 | Location ID doesn't exist |
| `WEATHER_DATA_UNAVAILABLE` | 503 | Weather API unavailable, retry later |
| `QUOTA_EXCEEDED` | 429 | Rate limit exceeded |
| `SIMULATION_QUEUE_FULL` | 503 | Too many queued simulations |
| `UNAUTHORIZED` | 401 | Missing/invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected error |

---

## 🔐 AUTHENTICATION & AUTHORIZATION

### Token-Based (JWT)

```python
# Endpoint: POST /auth/login
# Request: {"email": "user@example.com", "password": "..."}
# Response: {"access_token": "eyJhbGc...", "token_type": "bearer", "expires_in": 3600}

# All subsequent requests:
# Headers: {"Authorization": "Bearer eyJhbGc..."}
```

### Roles

```
anon      → Can view public locations, run limited simulations
user      → Full simulation access, can save buildings
admin     → Can manage locations, scenarios, data curation
data      → Can ingest weather data, manage refreshes
```

---

## 📊 RATE LIMITING

```
Tier              | Requests/hour | Simulations/day | Storage
──────────────────┼───────────────┼─────────────────┼────────────
Free              | 100           | 5               | 100 MB
Pro               | 1,000         | 100             | 10 GB
Enterprise        | 10,000        | Unlimited       | Unlimited
```

**Headers in Response**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 982
X-RateLimit-Reset: 1713607200
```

---

