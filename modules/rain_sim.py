"""
Rainfall Simulation Module
Generates stochastic daily rainfall data using probability and distribution models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import config


class RainfallSimulator:
    """
    Generates realistic daily rainfall data using stochastic models.
    
    Features:
    - Probability-based rain occurrence (30% default)
    - Intensity distribution based on gamma distribution
    - Reproducible results with seed control
    """
    
    def __init__(self, seed: int = config.RAIN_SEED):
        """
        Initialize the rainfall simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.rain_probability = config.RAIN_PROBABILITY
        self.gamma_shape = config.RAIN_GAMMA_SHAPE
        self.gamma_scale = config.RAIN_GAMMA_SCALE
        self.daily_rainfall = None
        
    def generate_annual_rainfall(self, days: int = config.SIMULATION_DAYS) -> np.ndarray:
        """
        Generate rainfall data for a full year (365 days).
        
        Algorithm:
        1. For each day, generate a random number [0,1]
        2. If random < rain_probability, rain occurs
        3. Sample intensity from gamma distribution
        4. Otherwise, rainfall = 0
        
        Args:
            days: Number of days to simulate (default 365)
            
        Returns:
            Array of daily rainfall in mm
        """
        # Step 1: Determine rain occurrence for each day
        rain_occurs = self.rng.random(days) < self.rain_probability
        
        # Step 2: Generate rainfall intensity for rainy days using gamma distribution
        # Gamma distribution produces realistic rainfall amounts
        rainfall_amounts = np.zeros(days)
        
        num_rainy_days = np.sum(rain_occurs)
        if num_rainy_days > 0:
            # Generate intensities only for days with rain
            intensities = self.rng.gamma(
                shape=self.gamma_shape,
                scale=self.gamma_scale,
                size=num_rainy_days
            )
            rainfall_amounts[rain_occurs] = intensities
        
        self.daily_rainfall = rainfall_amounts
        return rainfall_amounts
    
    def get_rainfall_on_day(self, day: int) -> float:
        """
        Get rainfall amount for a specific day.
        
        Args:
            day: Day number (0-364)
            
        Returns:
            Rainfall in mm
        """
        if self.daily_rainfall is None:
            self.generate_annual_rainfall()
        
        if day < 0 or day >= len(self.daily_rainfall):
            raise ValueError(f"Day must be between 0 and {len(self.daily_rainfall)-1}")
        
        return float(self.daily_rainfall[day])
    
    def get_statistics(self) -> dict:
        """
        Calculate and return rainfall statistics for the year.
        
        Returns:
            Dictionary with statistical measures
        """
        if self.daily_rainfall is None:
            self.generate_annual_rainfall()
        
        rainy_days = np.sum(self.daily_rainfall > 0)
        
        stats = {
            'total_rainfall': float(np.sum(self.daily_rainfall)),
            'average_daily': float(np.mean(self.daily_rainfall)),
            'max_daily': float(np.max(self.daily_rainfall)),
            'min_daily': float(np.min(self.daily_rainfall)),
            'rainy_days': int(rainy_days),
            'rain_probability_actual': float(rainy_days / len(self.daily_rainfall)),
            'std_dev': float(np.std(self.daily_rainfall))
        }
        
        return stats
    
    def get_monthly_summary(self) -> pd.DataFrame:
        """
        Calculate monthly rainfall summary.
        
        Returns:
            DataFrame with monthly statistics
        """
        if self.daily_rainfall is None:
            self.generate_annual_rainfall()
        
        # Create date range for the year
        dates = pd.date_range(start='2024-01-01', periods=len(self.daily_rainfall), freq='D')
        df = pd.DataFrame({
            'date': dates,
            'rainfall_mm': self.daily_rainfall
        })
        
        # Group by month and calculate statistics
        df['month'] = df['date'].dt.month
        monthly = df.groupby('month').agg({
            'rainfall_mm': ['sum', 'mean', 'max', 'count']
        }).round(2)
        
        return monthly
    
    def reset_with_new_seed(self, seed: int):
        """
        Reset the simulator with a new random seed.
        
        Args:
            seed: New random seed
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.daily_rainfall = None

    def load_external(self, rainfall_mm) -> np.ndarray:
        """
        Inject an externally provided daily rainfall series (e.g. from
        Open-Meteo), bypassing stochastic generation.

        Args:
            rainfall_mm: 1D iterable of daily rainfall in millimeters.

        Returns:
            The resulting numpy array stored as ``daily_rainfall``.
        """
        arr = np.asarray(rainfall_mm, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        self.daily_rainfall = arr
        return arr


def calculate_collected_water(
    rainfall_mm: float,
    roof_area_m2: float,
    efficiency: float = config.ROOF_EFFICIENCY
) -> float:
    """
    Calculate water collected from roof given rainfall.
    
    Formula: collected_water = rainfall × roof_area × efficiency
    
    Args:
        rainfall_mm: Daily rainfall in millimeters
        roof_area_m2: Roof area in square meters
        efficiency: Collection efficiency (0 to 1)
        
    Returns:
        Collected water in liters
    """
    # mm to liters/m² conversion: 1 mm = 1 liter/m²
    collected_liters = rainfall_mm * roof_area_m2 * config.MM_TO_LITERS_PER_M2 * efficiency
    return collected_liters
