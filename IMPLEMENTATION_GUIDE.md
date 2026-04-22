# 🚀 IMPLEMENTATION GUIDE: Quick Start for Development Teams

**Purpose**: Actionable tasks to begin Phase 1 immediately  
**Audience**: Development teams, tech leads  
**Duration**: ~8 months, 4 teams of 2 engineers

---

## PHASE 1: FOUNDATION (Weeks 1-4)

### Week 1: Project Infrastructure

#### Tasks (Priority: CRITICAL)

```python
# Task 1.1: Repository Setup
- [ ] Initialize Git repo with conventional commits
- [ ] Create branch strategy: main → develop → feature/**
- [ ] Add .gitignore (Python, environment, secrets)
- [ ] Create CONTRIBUTING.md
Code:
  # .gitignore essentials
  venv/
  __pycache__/
  .env
  .env.local
  *.pyc
  .pytest_cache/
  .coverage
  htmlcov/
  .DS_Store
  .vscode/settings.json  # local editor config
  dist/
  build/
  *.egg-info/

# Task 1.2: Poetry/Pip Setup
- [ ] Choose: Poetry (recommended) vs pip + venv
- [ ] Create pyproject.toml or requirements.txt
- [ ] Dev dependencies: pytest, black, flake8, mypy, pre-commit
- [ ] Lock dependencies (poetry.lock or requirements-lock.txt)
Code:
  [project]
  name = "rainwater-harvesting-platform"
  version = "2.0.0"
  description = "Geospatial rainwater harvesting analysis"
  
  [dependencies]
  streamlit = "^1.28.0"
  fastapi = "^0.100.0"
  sqlalchemy = "^2.0.0"
  pandas = "^2.0.0"
  numpy = "^1.24.0"
  scipy = "^1.10.0"
  pydantic = "^2.0.0"
  redis = "^5.0.0"
  httpx = "^0.24.0"  # async HTTP client
  meteostat = "^1.6.0"  # or implement custom API client

# Task 1.3: Docker Development Environment
- [ ] Create docker-compose.yml with services:
  - Backend (FastAPI)
  - PostgreSQL + PostGIS
  - InfluxDB
  - Redis
  - pgAdmin (database UI)
Code:
  version: '3.8'
  services:
    postgres:
      image: postgis/postgis:14-3.3
      environment:
        POSTGRES_USER: dev
        POSTGRES_PASSWORD: devpass
        POSTGRES_DB: rainwater_dev
      ports:
        - "5432:5432"
      volumes:
        - postgres_data:/var/lib/postgresql/data
    
    redis:
      image: redis:7-alpine
      ports:
        - "6379:6379"
    
    influxdb:
      image: influxdb:2.7-alpine
      environment:
        INFLUXDB_DB: rainwater
        INFLUXDB_ADMIN_USER: admin
        INFLUXDB_ADMIN_PASSWORD: admin
      ports:
        - "8086:8086"
  
  volumes:
    postgres_data:

# Task 1.4: Logging & Configuration
- [ ] Implement structured logging (Python logging + JSON formatter)
- [ ] Create config management (Pydantic Settings with .env)
- [ ] Add logging.yaml for log levels per module
Code:
  # backend/config/settings.py
  from pydantic_settings import BaseSettings
  
  class Settings(BaseSettings):
      # Database
      DATABASE_URL: str = "postgresql://dev:devpass@localhost:5432/rainwater_dev"
      
      # Redis
      REDIS_URL: str = "redis://localhost:6379"
      
      # API
      API_TITLE: str = "Rainwater Harvesting Platform"
      DEBUG: bool = True
      
      # Simulation
      SIMULATION_MAX_WORKERS: int = 4
      
      class Config:
          env_file = ".env"
  
  settings = Settings()

- [ ] Create logging factory:
  
  # backend/config/logging_config.py
  import logging
  import json
  from pythonjsonlogger import jsonlogger
  
  def setup_logging():
      logger = logging.getLogger()
      logHandler = logging.StreamHandler()
      formatter = jsonlogger.JsonFormatter()
      logHandler.setFormatter(formatter)
      logger.addHandler(logHandler)
      logger.setLevel(logging.INFO)
      return logger

# Task 1.5: Pre-commit Hooks
- [ ] Set up pre-commit hooks for code quality
Code:
  # .pre-commit-config.yaml
  repos:
    - repo: https://github.com/psf/black
      rev: 23.7.0
      hooks:
        - id: black
          language_version: python3.11
    
    - repo: https://github.com/pycqa/flake8
      rev: 6.0.0
      hooks:
        - id: flake8
    
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.4.1
      hooks:
        - id: mypy
          additional_dependencies: [types-all]

  # Install:
  # $ pre-commit install
```

---

### Week 2: Database Setup & Data Layer

#### Tasks (Priority: CRITICAL)

```python
# Task 2.1: Database Schema Design
- [ ] Create ERD (Entity Relationship Diagram)
- [ ] Design tables:
  
  a) Users & Sessions
     users (id, email, created_at, updated_at)
     sessions (id, user_id, token, expires_at)
  
  b) Locations & Geography
     locations (id, latitude, longitude, name, country, created_at)
     admin_boundaries (id, name, geometry, level)  -- region/country/city
  
  c) Buildings & Portfolio
     buildings (id, user_id, roof_area_m2, efficiency, height_m, geometry, created_at)
     building_portfolios (id, user_id, name, description, buildings)
  
  d) Weather & Climate
     weather_stations (id, location_id, station_code, data_source)
     daily_precipitation (id, station_id, date, precip_mm, temp_c, humidity_pct)
     weather_profiles (id, location_id, p_rain_annual, gamma_shape, gamma_scale, ...)
  
  e) Simulations & Results
     simulation_configs (id, building_id, roof_area, efficiency, tank_capacity, ...)
     simulation_results (id, config_id, date_run, annual_yield_m3, reliability_pct, ...)
     daily_simulation_state (id, result_id, day_of_year, tank_level_L, collected_L, ...)
  
  f) Scenarios & Analytics
     scenarios (id, name, precip_change_pct, intensity_change_pct, created_at)
     scenario_results (id, scenario_id, config_id, result_json)

# Task 2.2: SQLAlchemy ORM Models
- [ ] Implement models for all tables
Code:
  # data/database/models.py
  from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
  from sqlalchemy.ext.declarative import declarative_base
  from datetime import datetime
  
  Base = declarative_base()
  
  class Location(Base):
      __tablename__ = "locations"
      
      id = Column(Integer, primary_key=True, index=True)
      latitude = Column(Float, nullable=False)
      longitude = Column(Float, nullable=False)
      name = Column(String(255))
      country = Column(String(100))
      created_at = Column(DateTime, default=datetime.utcnow)
      
  class Building(Base):
      __tablename__ = "buildings"
      
      id = Column(Integer, primary_key=True, index=True)
      user_id = Column(Integer, ForeignKey("users.id"))
      location_id = Column(Integer, ForeignKey("locations.id"))
      roof_area_m2 = Column(Float)
      efficiency = Column(Float)  # 0.7-0.95
      height_m = Column(Float, nullable=True)
      created_at = Column(DateTime, default=datetime.utcnow)
  
  class DailyPrecipitation(Base):
      __tablename__ = "daily_precipitation"
      
      id = Column(Integer, primary_key=True, index=True)
      location_id = Column(Integer, ForeignKey("locations.id"))
      date = Column(DateTime, nullable=False)
      precip_mm = Column(Float)
      created_at = Column(DateTime, default=datetime.utcnow)
      
      __table_args__ = (
          UniqueConstraint('location_id', 'date', name='unique_location_date'),
      )

# Task 2.3: Alembic Migrations Setup
- [ ] Initialize Alembic:
  $ alembic init alembic
  
- [ ] Configure alembic.ini to use SQLAlchemy env
- [ ] Create first migration:
  $ alembic revision --autogenerate -m "Initial schema"
  
- [ ] Apply migration:
  $ alembic upgrade head

# Task 2.4: Data Loaders (Stubs)
- [ ] Create loader interfaces:
Code:
  # data/loaders/base_loader.py
  from abc import ABC, abstractmethod
  import pandas as pd
  
  class BaseWeatherLoader(ABC):
      @abstractmethod
      async def fetch_daily_precipitation(
          self,
          latitude: float,
          longitude: float,
          start_date: str,
          end_date: str
      ) -> pd.DataFrame:
          """Return DataFrame with columns: date, precip_mm"""
          pass
  
  # data/loaders/meteostat_loader.py
  class MeteostatLoader(BaseWeatherLoader):
      async def fetch_daily_precipitation(self, ...):
          # Placeholder for now
          # Week 3 will implement actual API calls
          pass

# Task 2.5: Database Utilities
- [ ] Connection factory (SQLAlchemy + connection pooling)
Code:
  # data/database/connection.py
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker, Session
  from backend.config.settings import settings
  
  engine = create_engine(
      settings.DATABASE_URL,
      pool_size=20,
      max_overflow=40,
      pool_pre_ping=True,  # Test connections before use
      echo=False  # Set True for SQL logging
  )
  
  SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
  
  def get_db() -> Session:
      db = SessionLocal()
      try:
          yield db
      finally:
          db.close()
```

---

### Week 3: API Layer Setup & Weather Integration

#### Tasks (Priority: HIGH)

```python
# Task 3.1: FastAPI Project Structure
- [ ] Create main.py with app initialization
Code:
  # backend/main.py
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from backend.routers import location, weather, simulation, buildings
  from backend.config import settings
  
  app = FastAPI(
      title=settings.API_TITLE,
      description="Geospatial Rainwater Harvesting Platform API",
      version="2.0.0"
  )
  
  # CORS
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # Replace with specific origins
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  
  # Routes
  app.include_router(location.router, prefix="/api/v1/location")
  app.include_router(weather.router, prefix="/api/v1/weather")
  app.include_router(simulation.router, prefix="/api/v1/simulation")
  app.include_router(buildings.router, prefix="/api/v1/buildings")
  
  @app.get("/health")
  def health_check():
      return {"status": "ok"}

# Task 3.2: Pydantic Models (Request/Response)
- [ ] Define request/response schemas
Code:
  # backend/models/location.py
  from pydantic import BaseModel, Field
  from datetime import datetime
  
  class LocationRequest(BaseModel):
      latitude: float = Field(..., ge=-90, le=90)
      longitude: float = Field(..., ge=-180, le=180)
      location_name: str = Field(...)
  
  class LocationResponse(BaseModel):
      id: int
      latitude: float
      longitude: float
      location_name: str
      country: str
      created_at: datetime
      
      class Config:
          from_attributes = True

# Task 3.3: Middleware (Logging, Error Handling)
- [ ] Add request/response logging middleware
Code:
  # backend/middleware/logging.py
  from starlette.middleware.base import BaseHTTPMiddleware
  import logging
  import json
  
  class LoggingMiddleware(BaseHTTPMiddleware):
      def __init__(self, app, logger: logging.Logger):
          super().__init__(app)
          self.logger = logger
      
      async def dispatch(self, request, call_next):
          # Log request
          self.logger.info(f"-> {request.method} {request.url.path}")
          
          response = await call_next(request)
          
          # Log response
          self.logger.info(f"<- {response.status_code}")
          
          return response

- [ ] Add error handler middleware:
Code:
  # backend/middleware/error_handler.py
  from fastapi import Request
  from fastapi.responses import JSONResponse
  
  @app.exception_handler(Exception)
  async def general_exception_handler(request: Request, exc: Exception):
      logger.error(f"Unhandled exception: {exc}", exc_info=True)
      return JSONResponse(
          status_code=500,
          content={"detail": "Internal server error"}
      )

# Task 3.4: Meteostat API Client
- [ ] Implement Meteostat integration (basic version)
Code:
  # backend/services/weather_service.py
  from meteostat import Point, Daily
  import pandas as pd
  from datetime import datetime, timedelta
  import logging
  
  class MeteostatWeatherService:
      def __init__(self, cache_service, logger=None):
          self.cache = cache_service
          self.logger = logger or logging.getLogger(__name__)
      
      async def fetch_daily_precipitation(
          self,
          latitude: float,
          longitude: float,
          start_date: str = "2014-01-01",
          end_date: str = None
      ) -> pd.DataFrame:
          """
          Fetch daily precipitation from Meteostat.
          
          Return DataFrame with columns: date, prcp (mm)
          """
          if end_date is None:
              end_date = datetime.now().strftime("%Y-%m-%d")
          
          # Try cache first
          cache_key = f"meteo_{latitude}_{longitude}_{start_date}_{end_date}"
          cached = await self.cache.get(cache_key)
          if cached:
              self.logger.info(f"Cache hit: {cache_key}")
              return pd.read_json(cached)
          
          try:
              # Query Meteostat
              point = Point(latitude, longitude)
              start = datetime.strptime(start_date, "%Y-%m-%d")
              end = datetime.strptime(end_date, "%Y-%m-%d")
              
              data = Daily(point, start, end).fetch()
              
              if data is None or data.empty:
                  raise ValueError("No data returned from Meteostat")
              
              # Extract precipitation (prcp column)
              result = data[['prcp']].reset_index()
              result.columns = ['date', 'precip_mm']
              
              # Cache for 30 days
              await self.cache.set_json(cache_key, result, ttl=2592000)
              
              self.logger.info(f"Fetched {len(result)} days from Meteostat")
              return result
          
          except Exception as e:
              self.logger.error(f"Meteostat fetch failed: {e}")
              raise

# Task 3.5: Route Stubs
- [ ] Create routers with endpoint stubs
Code:
  # backend/routers/location.py
  from fastapi import APIRouter, Depends
  from backend.models.location import LocationRequest, LocationResponse
  from backend.services.location_service import LocationService
  from backend.data.database.connection import get_db
  
  router = APIRouter()
  
  @router.post("/select", response_model=LocationResponse)
  async def select_location(
      request: LocationRequest,
      db = Depends(get_db)
  ):
      """User selects a location from map."""
      service = LocationService(db)
      location = await service.save_location(request)
      return location
  
  @router.get("/{location_id}", response_model=LocationResponse)
  async def get_location(location_id: int, db = Depends(get_db)):
      """Retrieve location details."""
      # Implement in Task 3.6
      pass
```

---

### Week 4: Statistical Rainfall Model (Offline)

#### Tasks (Priority: HIGH)

```python
# Task 4.1: Rainfall Distribution Fitting
- [ ] Implement Bernoulli + Gamma parameter fitting
Code:
  # core/rainfall_model/distribution_fitter.py
  import numpy as np
  from scipy.stats import gamma, kstest
  import pandas as pd
  
  class RainfallDistributionFitter:
      def __init__(self, daily_precip_mm):
          """
          Args:
              daily_precip_mm: np.array of 365+ daily values
          """
          self.daily_data = daily_precip_mm
          self.rain_threshold = 0.1  # mm
      
      def fit_bernoulli(self, seasonal=False):
          """
          Estimate probability of rain (P > threshold).
          
          Returns:
              If seasonal=False: float (0-1)
              If seasonal=True: np.array of 12 values (monthly)
          """
          binary_rain = (self.daily_data > self.rain_threshold).astype(int)
          
          if not seasonal:
              # Annual average
              p_rain = binary_rain.mean()
              return p_rain
          else:
              # Monthly averages (assume 365 days = 12 × ~30.4 days)
              p_rain_monthly = []
              days_per_month = len(self.daily_data) / 12
              
              for month in range(12):
                  start_idx = int(month * days_per_month)
                  end_idx = int((month + 1) * days_per_month)
                  p = binary_rain[start_idx:end_idx].mean()
                  p_rain_monthly.append(p)
              
              return np.array(p_rain_monthly)
      
      def fit_gamma(self, seasonal=False):
          """
          Fit gamma distribution to rainy days only.
          
          Returns:
              If seasonal=False: (shape, loc, scale)
              If seasonal=True: list of 12 tuples
          """
          rainy_days = self.daily_data[self.daily_data > self.rain_threshold]
          
          if len(rainy_days) < 10:
              raise ValueError("Insufficient rainy days for fitting")
          
          if not seasonal:
              shape, loc, scale = gamma.fit(rainy_days)
              return shape, loc, scale
          else:
              params_monthly = []
              days_per_month = len(self.daily_data) / 12
              
              for month in range(12):
                  start_idx = int(month * days_per_month)
                  end_idx = int((month + 1) * days_per_month)
                  rainy_month = self.daily_data[start_idx:end_idx]
                  rainy_month = rainy_month[rainy_month > self.rain_threshold]
                  
                  if len(rainy_month) > 2:
                      shape, loc, scale = gamma.fit(rainy_month)
                      params_monthly.append((shape, loc, scale))
                  else:
                      params_monthly.append((1.0, 0.0, 5.0))  # Default
              
              return params_monthly
      
      def validate_fit(self, shape, scale, n_bootstrap=100):
          """
          Goodness-of-fit testing (Kolmogorov-Smirnov).
          
          Returns:
              p_value: If > 0.05, fit is good
          """
          # Generate synthetic from fitted distribution
          synthetic = np.random.gamma(shape, scale, n_bootstrap)
          rainy_observed = self.daily_data[self.daily_data > self.rain_threshold]
          
          # KS test
          statistic, p_value = kstest(synthetic, 'gamma', args=(shape, 0, scale))
          return p_value

# Task 4.2: Stochastic Rainfall Generator
- [ ] Implement synthetic rainfall generation
Code:
  # core/rainfall_model/stochastic_generator.py
  import numpy as np
  
  class StochasticRainfallGenerator:
      def __init__(self, p_rain, gamma_params, rain_threshold=0.1):
          """
          Args:
              p_rain: Bernoulli parameter (scalar or 12-element array)
              gamma_params: Gamma parameters (scalar tuple or 12 tuples)
                           Each tuple: (shape, loc, scale)
              rain_threshold: mm, below which consider "no rain"
          """
          self.p_rain = p_rain
          self.gamma_params = gamma_params
          self.rain_threshold = rain_threshold
      
      def generate_series(self, n_days=365, seed=42, seasonal=True):
          """
          Generate synthetic daily rainfall time-series.
          
          Args:
              n_days: simulation horizon (typically 365)
              seed: int for reproducibility
              seasonal: use monthly parameters?
          
          Returns:
              np.array of shape (n_days,) with daily precip in mm
          """
          np.random.seed(seed)
          rainfall = np.zeros(n_days)
          
          for day in range(n_days):
              # Determine which month this is
              if seasonal:
                  month = int((day / n_days) * 12) % 12
                  p = self.p_rain[month]
                  shape, loc, scale = self.gamma_params[month]
              else:
                  p = self.p_rain
                  shape, loc, scale = self.gamma_params
              
              # Bernoulli: is it rainy?
              if np.random.rand() < p:
                  # Yes: sample from gamma
                  rainfall[day] = np.random.gamma(shape, scale)
              else:
                  # No: zero precipitation
                  rainfall[day] = 0.0
          
          return rainfall

# Task 4.3: Model Validation Tests
- [ ] Unit tests for distribution fitting
Code:
  # tests/unit/test_rainfall_model.py
  import pytest
  import numpy as np
  from core.rainfall_model.distribution_fitter import RainfallDistributionFitter
  from core.rainfall_model.stochastic_generator import StochasticRainfallGenerator
  
  @pytest.fixture
  def sample_precip_data():
      """365 days of synthetic precipitation."""
      np.random.seed(42)
      return np.random.gamma(2.0, 5.0, 365) * np.random.binomial(1, 0.3, 365)
  
  def test_bernoulli_fit(sample_precip_data):
      fitter = RainfallDistributionFitter(sample_precip_data)
      p_rain = fitter.fit_bernoulli(seasonal=False)
      
      assert 0 <= p_rain <= 1
      assert isinstance(p_rain, (float, np.floating))
  
  def test_gamma_fit(sample_precip_data):
      fitter = RainfallDistributionFitter(sample_precip_data)
      shape, loc, scale = fitter.fit_gamma(seasonal=False)
      
      assert shape > 0
      assert scale > 0
  
  def test_stochastic_generation(sample_precip_data):
      fitter = RainfallDistributionFitter(sample_precip_data)
      p_rain = fitter.fit_bernoulli()
      shape, loc, scale = fitter.fit_gamma()
      
      generator = StochasticRainfallGenerator(p_rain, (shape, loc, scale))
      synthetic = generator.generate_series(365, seed=42)
      
      assert len(synthetic) == 365
      assert np.all(synthetic >= 0)
      assert synthetic.mean() > 0

# Task 4.4: Pre-computed Model Parameters
- [ ] Pre-fit models for 5-10 test cities
- [ ] Store in database (weather_profiles table)
- [ ] Create script to batch-fit locations

Code:
  # scripts/batch_fit_models.py
  """
  Pre-fit rainfall models for test locations.
  Run once, results stored in database.
  """
  import asyncio
  import pandas as pd
  from backend.services.weather_service import MeteostatWeatherService
  from core.rainfall_model.distribution_fitter import RainfallDistributionFitter
  from data.database.models import WeatherProfile
  from data.database.connection import SessionLocal
  
  TEST_LOCATIONS = [
      {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784, "country": "Turkey"},
      {"name": "Precipitation", "lat": 35.6762, "lon": 139.6503, "country": "Japan"},
      # Add more locations
  ]
  
  async def batch_fit():
      cache = None  # Placeholder for cache service
      weather_svc = MeteostatWeatherService(cache)
      db = SessionLocal()
      
      for loc in TEST_LOCATIONS:
          print(f"Fitting {loc['name']}...")
          
          # Fetch 10 years of data
          data = await weather_svc.fetch_daily_precipitation(
              loc['lat'], loc['lon'],
              start_date="2014-01-01"
          )
          
          # Fit model
          precip_array = data['precip_mm'].values
          fitter = RainfallDistributionFitter(precip_array)
          
          p_rain_annual = fitter.fit_bernoulli(seasonal=False)
          p_rain_monthly = fitter.fit_bernoulli(seasonal=True)
          shape, loc_param, scale = fitter.fit_gamma(seasonal=False)
          
          # Store in database
          profile = WeatherProfile(
              location_name=loc['name'],
              p_rain_annual=float(p_rain_annual),
              p_rain_monthly=list(p_rain_monthly),
              gamma_shape=float(shape),
              gamma_scale=float(scale)
          )
          db.add(profile)
          db.commit()
      
      print("Batch fitting complete!")
  
  if __name__ == "__main__":
      asyncio.run(batch_fit())
```

---

## PHASE 2: SIMULATION ENGINE (Weeks 5-8)

### Week 5: Core Harvesting Model

Tasks (will provide in next section):

```python
# Task 5.1: Daily Collection Model
class HarvestingModel:
    def daily_collection(precip_mm, roof_area_m2, efficiency):
        # V = P × A × C formula
        pass
    
    def daily_consumption(workers, hours_worked, consumption_per_worker_per_hour):
        # Total daily water demand
        pass

# Task 5.2: Tank State Evolution
class TankSimulator:
    def update_state(self, collected_L, consumed_L):
        # Tank.level += collected - consumed
        # Handle constraints (capacity, negative level)
        pass

# Task 5.3: Single 365-day Run
class SimulationRunner:
    def run_single(self, rainfall_series, parameters):
        # Daily loop: t=0 to 364
        # Update tank, track metrics
        pass
```

---

## 🎯 IMMEDIATE NEXT STEPS (TODAY/THIS WEEK)

1. **Copy this checklist to your issue tracker** (GitHub Issues/Jira)
2. **Assign tasks to team members**:
   - Week 1: 1 engineer (infra/setup)
   - Week 2: 1 engineer (database)
   - Week 3: 1 engineer (API)
   - Week 4: 1 engineer (modeling)
3. **Set up daily standups** (15 min, async updates acceptable)
4. **Create Milestone markers** for each week
5. **Begin Week 1 tasks TODAY**

---

## 📊 Week 1 Definition of Done

- [ ] Project repo initialized + branch strategy in place
- [ ] Docker Compose running (all services healthy)
- [ ] Logging configured + test working
- [ ] Config management (.env pattern) working
- [ ] First pre-commit hook test passing
- [ ] README updated with setup instructions
- [ ] Daily standup scheduled for Week 2

**Estimated effort**: 40 hours (1 engineer full-time)

---

