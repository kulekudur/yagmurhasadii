"""
Configuration file for the Rainwater Harvesting Simulation Platform
Central place for all simulation parameters and constants
"""

# ===== SIMULATION PARAMETERS =====
SIMULATION_DAYS = 365
HOURS_PER_DAY = 24

# ===== RAINFALL MODEL PARAMETERS =====
# Stochastic rainfall generation
RAIN_PROBABILITY = 0.30  # Probability of rain on any given day (30%)
RAIN_GAMMA_SHAPE = 2.0   # Shape parameter for gamma distribution
RAIN_GAMMA_SCALE = 5.0   # Scale parameter for gamma distribution (mm)
RAIN_SEED = 42            # Random seed for reproducibility

# ===== ROOFTOP COLLECTION =====
ROOF_EFFICIENCY = 0.85    # Water collection efficiency (0-1)
ROOF_AREA_DEFAULT = 500   # Default roof area in m²

# ===== TANK MODEL =====
TANK_CAPACITY_DEFAULT = 50000  # Default tank capacity in liters
TANK_INITIAL_LEVEL = 0.5       # Initial tank level as % of capacity

# ===== HUMAN CONSUMPTION =====
WORKER_COUNT_DEFAULT = 50            # Default number of workers
CONSUMPTION_PER_WORKER_PER_HOUR = 2.0  # Liters per worker per hour
WORK_START_HOUR = 9                  # Start of working hours (24-hour format)
WORK_END_HOUR = 17                   # End of working hours (24-hour format)

# ===== ECONOMIC PARAMETERS =====
WATER_PRICE = 6.50          # Turkish Lira per m³ (per 1000 liters) - realistic municipal rate
TANK_COST = 5000            # Tank capex in TL (depo maliyeti)
INSTALLATION_COST = 3500    # Boru, pompa, filtre, first-flush, işçilik
MAINTENANCE_COST = 500      # Yıllık bakım gideri (TL)

# Discounted Payback / DCF ekonomik varsayımları (reel terimler)
WATER_PRICE_ESCALATION = 0.03  # Su tarifesinin reel yıllık artışı (genel enflasyon üstü)
DISCOUNT_RATE = 0.05           # Reel ağırlıklı sermaye maliyeti (yatırımcı iskonto oranı)
SYSTEM_LIFESPAN_YEARS = 20     # Sistem/depo tipik ekonomik ömrü

# ===== VISUALIZATION PARAMETERS =====
BUILDING_WIDTH = 20
BUILDING_HEIGHT = 15
BUILDING_DEPTH = 15

TANK_RADIUS = 3
TANK_HEIGHT = 8

RAIN_PARTICLE_COUNT_MAX = 1000
PARTICLE_SIZE = 0.1

# ===== UI COLORS =====
COLOR_BUILDING = "#B0C4DE"
COLOR_TANK = "#4682B4"
COLOR_WATER = "#1E90FF"
COLOR_RAIN = "#87CEEB"
COLOR_WORKER = "#FF6347"

# ===== PATHS =====
ASSETS_PATH = "assets"
RESULTS_EXPORT_PATH = "results"

# ===== UNIT CONVERSIONS =====
MM_TO_LITERS_PER_M2 = 1.0  # 1 mm rainfall = 1 liter/m²
