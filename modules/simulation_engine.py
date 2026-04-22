"""
Main Simulation Engine
Orchestrates all components: rainfall, tank, workers, and economy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import config
from modules.rain_sim import RainfallSimulator, calculate_collected_water
from modules.tank_sim import StorageTank
from modules.human_sim import WorkforceSimulator
from modules.economy import EconomicAnalyzer


def replay_tank_for_rainfall(
    rainfall_mm: np.ndarray,
    roof_area: float,
    efficiency: float,
    tank_capacity: float,
    daily_consumption_liters: float,
    initial_level_pct: float = config.TANK_INITIAL_LEVEL * 100,
) -> Dict[str, np.ndarray]:
    """
    Deterministic day-by-day replay of a rainwater-harvesting tank.

    Runs independently from ``SimulationEngine`` so it can be used to drive
    animations from either synthetic or real (Open-Meteo) rainfall arrays.

    Args:
        rainfall_mm: 1D array of daily rainfall in millimeters.
        roof_area: Collection area in m^2.
        efficiency: Collection efficiency in [0, 1].
        tank_capacity: Tank capacity in liters.
        daily_consumption_liters: Liters consumed per day (flat model).
        initial_level_pct: Starting tank fill in % of capacity.

    Returns:
        Dict with numpy arrays (same length as rainfall_mm):
            - ``levels_l``: tank level in liters at end of each day
            - ``levels_pct``: tank fill percentage at end of each day
            - ``inflow_l``: liters collected that day
            - ``outflow_l``: liters actually drawn (capped by availability)
    """
    rainfall_mm = np.asarray(rainfall_mm, dtype=float)
    n = len(rainfall_mm)

    levels_l = np.zeros(n, dtype=float)
    levels_pct = np.zeros(n, dtype=float)
    inflow_l = np.zeros(n, dtype=float)
    outflow_l = np.zeros(n, dtype=float)

    capacity = max(1e-6, float(tank_capacity))
    level = capacity * (max(0.0, min(100.0, initial_level_pct)) / 100.0)

    for i, mm in enumerate(rainfall_mm):
        inflow = max(0.0, float(mm)) * roof_area * config.MM_TO_LITERS_PER_M2 * efficiency
        demand = max(0.0, float(daily_consumption_liters))
        available = level + inflow
        actual_out = min(demand, available)
        level = max(0.0, min(capacity, available - actual_out))

        levels_l[i] = level
        levels_pct[i] = (level / capacity) * 100.0
        inflow_l[i] = inflow
        outflow_l[i] = actual_out

    return {
        'levels_l': levels_l,
        'levels_pct': levels_pct,
        'inflow_l': inflow_l,
        'outflow_l': outflow_l,
    }


class SimulationEngine:
    """
    Main simulation engine that runs the complete rainwater harvesting system.
    
    Integrates:
    - Rainfall generation
    - Tank dynamics
    - Water consumption
    - Economic analysis
    """
    
    def __init__(
        self,
        roof_area: float = config.ROOF_AREA_DEFAULT,
        roof_efficiency: float = config.ROOF_EFFICIENCY,
        tank_capacity: float = config.TANK_CAPACITY_DEFAULT,
        worker_count: int = config.WORKER_COUNT_DEFAULT,
        rain_seed: int = config.RAIN_SEED,
        external_rainfall: Optional[np.ndarray] = None,
    ):
        """
        Initialize the simulation engine.

        Args:
            roof_area: Collection area in m²
            roof_efficiency: Collection efficiency (0-1)
            tank_capacity: Tank capacity in liters
            worker_count: Number of workers
            rain_seed: Random seed for reproducibility
            external_rainfall: Optional 1D array of daily rainfall (mm). When
                provided it replaces stochastic generation so downstream
                metrics reflect real-world data. Falls back to stochastic
                generation when ``None``.
        """
        # Initialize components
        self.rainfall_sim = RainfallSimulator(seed=rain_seed)
        self.tank = StorageTank(capacity=tank_capacity)
        self.workforce = WorkforceSimulator(worker_count=worker_count)
        self.economy = EconomicAnalyzer()

        # Simulation parameters
        self.roof_area = roof_area
        self.roof_efficiency = roof_efficiency
        self.worker_count = worker_count
        self.rain_seed = rain_seed
        self.rainfall_source = 'stochastic'

        # Simulation state
        self.current_day = 0
        self.current_hour = 0
        self.simulation_complete = False

        # History tracking
        self.daily_history = {
            'day': [],
            'rainfall': [],
            'collected': [],
            'consumed': [],
            'tank_level': [],
            'water_shortage': []
        }

        # Rainfall: prefer external series when given, otherwise generate.
        if external_rainfall is not None and len(external_rainfall) > 0:
            loaded = self.rainfall_sim.load_external(external_rainfall)
            self.rainfall_data = loaded
            self.rainfall_source = 'external'
        else:
            self.rainfall_data = self.rainfall_sim.generate_annual_rainfall()
    
    def run_full_simulation(self) -> Dict:
        """
        Run complete year-long simulation (365 days).
        
        Returns:
            Dictionary with simulation results
        """
        print("Starting full simulation...")
        
        for day in range(config.SIMULATION_DAYS):
            self.current_day = day
            self.run_daily_simulation()
            
            if (day + 1) % 50 == 0:
                print(f"  Simulated {day + 1}/{config.SIMULATION_DAYS} days")
        
        self.simulation_complete = True
        print("Simulation complete!")
        
        return self.get_simulation_results()
    
    def run_daily_simulation(self):
        """
        Run simulation for one day (24 hours).
        """
        day = self.current_day
        
        # Get rainfall for this day
        rainfall_mm = self.rainfall_sim.get_rainfall_on_day(day)
        
        # Calculate water collected from roof
        water_collected = calculate_collected_water(
            rainfall_mm=rainfall_mm,
            roof_area_m2=self.roof_area,
            efficiency=self.roof_efficiency
        )
        
        # Simulate each hour of the day
        daily_consumption = 0
        self.workforce.reset_daily()
        
        for hour in range(config.HOURS_PER_DAY):
            # Get hourly consumption for this hour
            hourly_consumption = self.workforce.get_hourly_consumption(hour)
            daily_consumption += hourly_consumption
            
            # For simplicity, distribute rainfall evenly over first hours of day
            hourly_inflow = water_collected / max(1, config.HOURS_PER_DAY)
            
            # Update tank state
            tank_level, shortage = self.tank.update(
                inflow=hourly_inflow,
                outflow=hourly_consumption
            )
        
        # Get daily totals from workforce
        daily_consumption_total = self.workforce.get_daily_consumption()
        shortage_occurred = daily_consumption_total > water_collected
        
        # Record in history
        self.daily_history['day'].append(day)
        self.daily_history['rainfall'].append(rainfall_mm)
        self.daily_history['collected'].append(water_collected)
        self.daily_history['consumed'].append(daily_consumption_total)
        self.daily_history['tank_level'].append(self.tank.get_current_level())
        self.daily_history['water_shortage'].append(shortage_occurred)
    
    def get_current_state(self) -> Dict:
        """
        Get current simulation state for a specific day/hour.
        
        Returns:
            Dictionary with current state
        """
        if not self.daily_history['day']:
            return {
                'day': 0,
                'hour': 0,
                'rainfall': 0,
                'collected': 0,
                'consumed': 0,
                'tank_level': 0,
                'tank_capacity': self.tank.get_capacity(),
                'tank_percentage': 0
            }
        
        day_idx = min(self.current_day, len(self.daily_history['day']) - 1)
        
        return {
            'day': self.daily_history['day'][day_idx],
            'hour': self.current_hour,
            'rainfall': self.daily_history['rainfall'][day_idx],
            'collected': self.daily_history['collected'][day_idx],
            'consumed': self.daily_history['consumed'][day_idx],
            'tank_level': self.daily_history['tank_level'][day_idx],
            'tank_capacity': self.tank.get_capacity(),
            'tank_percentage': self.tank.get_level_percentage()
        }
    
    def update_simulation_time(self, day: int, hour: int):
        """
        Update simulation to a specific day and hour.
        
        Args:
            day: Day number (0-364)
            hour: Hour (0-23)
        """
        day = max(0, min(day, config.SIMULATION_DAYS - 1))
        hour = max(0, min(hour, 23))
        
        self.current_day = day
        self.current_hour = hour
    
    def set_parameters(
        self,
        roof_area: float = None,
        roof_efficiency: float = None,
        tank_capacity: float = None,
        worker_count: int = None,
        rain_seed: int = None
    ):
        """
        Update simulation parameters and reset if necessary.
        
        Args:
            roof_area: New roof area
            roof_efficiency: New efficiency
            tank_capacity: New tank capacity
            worker_count: New worker count
            rain_seed: New random seed
        """
        if roof_area is not None:
            self.roof_area = roof_area
        
        if roof_efficiency is not None:
            self.roof_efficiency = roof_efficiency
        
        if tank_capacity is not None:
            self.tank.set_capacity(tank_capacity)
        
        if worker_count is not None:
            self.workforce.set_worker_count(worker_count)
            self.worker_count = worker_count
        
        if rain_seed is not None and rain_seed != self.rain_seed:
            self.rain_seed = rain_seed
            self.rainfall_sim.reset_with_new_seed(rain_seed)
            self.rainfall_data = self.rainfall_sim.generate_annual_rainfall()
            self.rainfall_source = 'stochastic'
    
    def get_simulation_results(self) -> Dict:
        """
        Get comprehensive simulation results and analysis.
        
        Returns:
            Dictionary with all results
        """
        if not self.daily_history['day']:
            return {}
        
        # Convert lists to numpy arrays for calculation
        collected = np.array(self.daily_history['collected'])
        consumed = np.array(self.daily_history['consumed'])
        tank_levels = np.array(self.daily_history['tank_level'])
        rainfall = np.array(self.daily_history['rainfall'])
        
        total_collected = float(np.sum(collected))
        total_consumed = float(np.sum(consumed))
        shortage_days = int(np.sum(self.daily_history['water_shortage']))
        
        # Get component statistics
        tank_stats = self.tank.get_statistics()
        rain_stats = self.rainfall_sim.get_statistics()
        workforce_stats = self.workforce.get_statistics()
        
        # Economic analysis
        economic_summary = self.economy.get_annual_summary(
            water_collected=total_collected,
            water_consumed=total_consumed,
            worker_count=self.workforce.get_worker_count()
        )
        
        return {
            'simulation_period': config.SIMULATION_DAYS,
            'water_metrics': {
                'total_collected': total_collected,
                'total_consumed': total_consumed,
                'utilization_rate': (total_consumed / total_collected * 100) if total_collected > 0 else 0,
                'shortage_days': shortage_days,
                'daily_average_collected': float(np.mean(collected)),
                'daily_average_consumed': float(np.mean(consumed))
            },
            'tank_metrics': tank_stats,
            'rainfall_metrics': rain_stats,
            'workforce_metrics': workforce_stats,
            'economic_metrics': economic_summary,
            'daily_history': self.daily_history,
            'system_parameters': {
                'roof_area_m2': self.roof_area,
                'roof_efficiency': self.roof_efficiency,
                'tank_capacity': self.tank.get_capacity(),
                'worker_count': self.workforce.get_worker_count()
            }
        }
    
    def export_results(self, filepath: str):
        """
        Export simulation results to CSV file.
        
        Args:
            filepath: Path to export to
        """
        import pandas as pd
        
        df = pd.DataFrame(self.daily_history)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
    
    def reset(self):
        """Reset simulation to initial state."""
        self.tank.reset()
        self.workforce.reset_daily()
        self.current_day = 0
        self.current_hour = 0
        self.simulation_complete = False
        self.daily_history = {
            'day': [],
            'rainfall': [],
            'collected': [],
            'consumed': [],
            'tank_level': [],
            'water_shortage': []
        }
        if self.rainfall_source == 'external' and self.rainfall_sim.daily_rainfall is not None:
            self.rainfall_data = self.rainfall_sim.daily_rainfall
        else:
            self.rainfall_data = self.rainfall_sim.generate_annual_rainfall()
