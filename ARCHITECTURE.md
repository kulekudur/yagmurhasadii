# Architecture & Design Documentation

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Dashboard Tabs: Overview | 3D | Graphs | Economics      │   │
│  │  Interactive Controls: Sliders, Time Navigation           │   │
│  │  Result Visualization & Export                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SIMULATION ENGINE (simulation_engine.py)            │
│                   Main Orchestration Layer                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Coordinates all modules                                 │   │
│  │  Manages daily/hourly simulation loop                    │   │
│  │  Tracks history and statistics                           │   │
│  │  Generates final results                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
    ▲            ▲              ▲              ▲            ▲
    │            │              │              │            │
    ▼            ▼              ▼              ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  RAIN    │ │  TANK    │ │ HUMAN    │ │ ECONOMY  │ │  VISUAL  │
│  MODULE  │ │ MODULE   │ │ MODULE   │ │ MODULE   │ │ MODULE   │
│          │ │          │ │          │ │          │ │          │
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│• Generate│ │• Track   │ │• Agent   │ │• ROI     │ │• 3D      │
│  rainfall│ │  level   │ │  behavior│ │  calc    │ │  scenes  │
│• Stoch   │ │• Model   │ │• Working │ │• Payback │ │• Graphs  │
│  model   │ │  inflow/ │ │  hours   │ │  period  │ │• Export  │
│• Stats   │ │  outflow │ │• Consum- │ │• Break   │ │  data    │
│          │ │• Shortage│ │  ption   │ │  even    │ │          │
│          │ │  tracking│ │• Profiles│ │• Savings │ │          │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Module Responsibilities

### 1. RainfallSimulator (rain_sim.py)

**Purpose**: Generate stochastic daily rainfall data

**Key Classes**:
```python
class RainfallSimulator:
    - generate_annual_rainfall()      # Main method
    - get_rainfall_on_day()           # Single day query
    - get_statistics()                # Annual stats
    - reset_with_new_seed()           # Reproducibility
```

**Algorithm**:
```
For each day in 365:
    1. Generate random [0,1]
    2. If random < rain_probability:
        - Sample intensity from Gamma(shape, scale)
        - Record rainfall
    else:
        - Record 0 mm
```

**Physics Model**:
```
Rainfall Generation:
  - Occurrence: Bernoulli(p=0.30)
  - Intensity: Gamma(α=2, β=5)
  
Collection:
  - Collected = Rainfall × Area × Efficiency
  - Unit conversion: 1mm = 1 L/m²
```

### 2. StorageTank (tank_sim.py)

**Purpose**: Model water tank dynamics with constraints

**Key Classes**:
```python
class StorageTank:
    - update()                        # Core state equation
    - get_current_level()             # Query state
    - get_level_percentage()          # Normalized level
    - can_supply()                    # Feasibility check
    - get_statistics()                # Performance metrics
```

**State Equation**:
```
tank(t+1) = max(0, min(tank(t) + inflow - outflow, capacity))

Constraints:
  - 0 ≤ level ≤ capacity
  - Shortage when: demand > available(collected + existing)
```

**Shortage Detection**:
```python
if current_level + inflow < outflow:
    water_shortage = True
    actual_outflow = current_level + inflow
    new_level = 0
```

### 3. WorkforceSimulator (human_sim.py)

**Purpose**: Model workers as time-dependent consumption agents

**Key Classes**:
```python
class WorkforceSimulator:
    - update_hour()                   # Set working status
    - get_hourly_consumption()        # Query consumption
    - get_daily_consumption()         # Daily total
    - get_consumption_profile()       # 24-hour pattern
```

**Consumption Model**:
```
For each hour:
    if work_start ≤ hour < work_end:
        consumption = workers × rate
    else:
        consumption = 0
        
Daily Total:
    consumption_day = Σ hourly_consumption
```

**Working Hours Logic**:
```python
def is_working_hour(hour):
    if work_start < work_end:
        return work_start ≤ hour < work_end
    else:  # Night shift wrap
        return hour ≥ work_start or hour < work_end
```

### 4. EconomicAnalyzer (economy.py)

**Purpose**: Financial analysis and ROI calculations

**Key Classes**:
```python
class EconomicAnalyzer:
    - calculate_roi()                 # Main metric
    - calculate_payback_period()      # Years to break-even
    - calculate_system_costs()        # Cost breakdown
    - get_breakeven_analysis()        # Viability check
    - sensitivity_analysis()          # Price sensitivity
```

**Financial Formulas**:
```
Cost Saved = water_consumed × water_price

Total Cost = installation + tank + (maintenance × years)

ROI% = ((savings - costs) / costs) × 100

Payback Years = total_cost / annual_savings

Break-Even = cost_saved ≥ total_cost
```

### 5. Scene3D (visualization.py)

**Purpose**: Create interactive 3D visualizations

**Key Classes**:
```python
class Scene3D:
    - create_building()               # Box mesh
    - create_tank()                   # Cylinder with fill
    - create_rain_particles()         # Fall effect
    - create_workers()                # Agent markers
    - create_full_scene()             # Composite render

class TimeSeriesGraphs:
    - create_tank_level_graph()       # Line chart
    - create_rainfall_graph()         # Bar chart
    - create_consumption_vs_supply()  # Dual line
```

**3D Components**:
```
Building:     Simple box (20m × 15m × 15m)
Tank:         Cylinder (r=3m, h=8m) with fill gradient
Rain:         Scattered particles (size ∝ intensity)
Workers:      Point markers at random indoor positions
```

### 6. SimulationEngine (simulation_engine.py)

**Purpose**: Orchestrate all modules and manage simulation flow

**Key Methods**:
```python
class SimulationEngine:
    - run_full_simulation()           # 365-day loop
    - run_daily_simulation()          # Daily update
    - get_current_state()             # Time-based query
    - get_simulation_results()        # Final analytics
    - set_parameters()                # Dynamic updates
    - export_results()                # CSV export
```

**Simulation Loop**:
```python
for day in 365:
    rainfall = rainfall_sim.get_rainfall_on_day(day)
    collected = rain_sim.calculate_collected_water(...)
    
    for hour in 24:
        consumption = workforce.get_hourly_consumption(hour)
        tank.update(hourly_inflow, consumption)
    
    record_daily_history(rainfall, collected, consumed, level)

generate_final_results()
```

## Data Flow

### Initialization Phase
```
Configuration (config.py)
        ▼
Create Engine:
  - RainfallSimulator (seed)
  - StorageTank (capacity)
  - WorkforceSimulator (workers)
  - EconomicAnalyzer (prices)
        ▼
Generate Rainfall Data (365 days precomputed)
```

### Simulation Phase
```
Day Loop (0-364):
  │
  ├─ Get Daily Rainfall
  │      ▼
  │  Calculate Collected Water
  │      ▼
  │  Hour Loop (0-23):
  │  │
  │  ├─ Get Worker Status (working?)
  │  │      ▼
  │  │  Get Hourly Consumption
  │  │      ▼
  │  │  Update Tank State
  │  │      ▼
  │  │  Record Hourly Data
  │  │
  │  └─ Next Hour
  │
  ├─ Record Daily Data
  │  - Daily rainfall
  │  - Total collected
  │  - Total consumed
  │  - Tank level
  │  - Shortage flag
  │
  └─ Next Day
```

### Analysis Phase
```
Generate Statistics:
  - Water metrics (collected, consumed, shortages)
  - Tank metrics (levels, efficiency)
  - Rainfall metrics (mean, max, distribution)
  - Workforce metrics (consumption patterns)
  - Economic metrics (ROI, payback, savings)

Generate Visualizations:
  - 3D scene with current state
  - Time-series graphs
  - Export data
```

## Configuration Hierarchy

```
Default Values (config.py)
        │
        ▼
Streamlit UI Input (app.py)
        │
        ▼
SimulationEngine Parameters
        │
        ├─ RainfallSimulator
        ├─ StorageTank
        ├─ WorkforceSimulator
        ├─ EconomicAnalyzer
        └─ Scene3D (visualization parameters)
```

## Extensibility Framework

### Adding New Module Type (e.g., Solar)

1. **Create Module** (modules/solar_sim.py):
```python
class SolarSimulator:
    def __init__(self, panel_area, efficiency):
        self.panel_area = panel_area
        self.efficiency = efficiency
    
    def generate_solar_data(self):
        # Generate solar irradiance
        pass
    
    def get_statistics(self):
        # Return metrics
        pass
```

2. **Integrate** (simulation_engine.py):
```python
class SimulationEngine:
    def __init__(self, ...):
        self.solar_sim = SolarSimulator(...)
    
    def run_daily_simulation(self):
        # ... existing code ...
        solar_energy = self.solar_sim.get_energy_on_day(day)
```

3. **Add UI** (app.py):
```python
with st.tab("☀️ Solar"):
    solar_result = engine.solar_sim.get_statistics()
    st.metric("Energy Generated", f"{solar_result} kWh")
```

### Plugin Architecture Concept

```
Future (v2.0):
├── BaseSimulationModule
│   ├── get_data()
│   ├── get_statistics()
│   └── reset()
│
├── RainfallSimulator (extends BaseSimulationModule)
├── SolarSimulator (extends BaseSimulationModule)
├── WindSimulator (extends BaseSimulationModule)
└── TrafficSimulator (extends BaseSimulationModule)

SimulationEngine:
    - Load plugins dynamically
    - Coordinate multiple modules
    - Combine results
```

## Performance Characteristics

### Time Complexity
```
Initialization:     O(1) - constant time
Daily Simulation:   O(24) = O(1) - 24 hours fixed
Annual Simulation:  O(365) = O(1) - constant days
Results Generation: O(1) - pre-computed statistics

Overall: O(1) - constant time for full simulation
```

### Space Complexity
```
Rainfall Data:      O(365) - annual data
Tank History:       O(365) - daily levels
Consumption Data:   O(365) - daily values
Workforce Agents:   O(workers) - typically ≤ 300

Overall: O(1) - constant space (small constants)
```

### Runtime Performance
```
Simulation:  ~1-2 seconds (365 days)
UI Load:     ~2-3 seconds
3D Render:   ~0.5-1 second per frame
Full App:    ~5-10 seconds total startup
```

## Error Handling

### Validation Layer
```
Input Validation:
  - roof_area > 0
  - 0 ≤ efficiency ≤ 1
  - tank_capacity > 0
  - worker_count ≥ 0
  - water_price > 0

State Constraints:
  - 0 ≤ tank_level ≤ capacity
  - 0 ≤ rainfall_mm
  - 0 ≤ consumption

Exception Handling:
  - Try-catch in simulation loop
  - Graceful degradation
```

## Testing Strategy

### Unit Tests (Per Module)
```python
test_rainfall_generation()
test_tank_constraints()
test_worker_working_hours()
test_economic_calculations()
test_visualization_rendering()
```

### Integration Tests
```python
test_full_simulation_consistency()
test_results_generation()
test_ui_data_flow()
test_export_functionality()
```

### Validation Tests
```python
test_water_balance()          # collected ≥ daily_stored
test_tank_capacity()          # level ≤ capacity
test_economic_consistency()   # savings relationships
```

## Future Enhancements

### Short Term (v1.1)
- [ ] Real weather data integration
- [ ] Batch parameter sweep
- [ ] Advanced export formats (Excel, PDF)
- [ ] System optimization algorithms

### Medium Term (v2.0)
- [ ] Plugin architecture
- [ ] Multi-building simulations
- [ ] Machine learning prediction
- [ ] Cloud deployment

### Long Term (v3.0)
- [ ] Distributed computing
- [ ] Real-time data streaming
- [ ] 3D model importing
- [ ] Advanced physics simulation

---

**Architecture Version**: 1.0  
**Design Paradigm**: Modular, extensible, composable  
**Status**: Production Ready
