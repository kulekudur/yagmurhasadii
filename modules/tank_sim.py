"""
Storage Tank Simulation Module
Models water tank dynamics with inflow (rainfall) and outflow (consumption)
"""

import numpy as np
from typing import List, Tuple
import config


class StorageTank:
    """
    Simulates a water storage tank with dynamic state updates.
    
    State equation: tank(t) = tank(t-1) + inflow - outflow
    Constraints: 0 ≤ tank_level ≤ capacity
    """
    
    def __init__(self, capacity: float = config.TANK_CAPACITY_DEFAULT):
        """
        Initialize the storage tank.
        
        Args:
            capacity: Tank capacity in liters
        """
        self.capacity = capacity
        self.current_level = capacity * config.TANK_INITIAL_LEVEL
        self.history_levels = [self.current_level]
        self.history_inflow = []
        self.history_outflow = []
        self.total_collected = 0
        self.total_consumed = 0
        self.days_insufficient = 0
        
    def update(self, inflow: float, outflow: float) -> Tuple[float, bool]:
        """
        Update tank state for one time step.
        
        Algorithm:
        1. Calculate new level: level_new = level_old + inflow - outflow
        2. Enforce constraints: 0 ≤ level_new ≤ capacity
        3. If outflow > available water, mark as water shortage
        
        Args:
            inflow: Water inflow to tank (liters)
            outflow: Water demand from tank (liters)
            
        Returns:
            Tuple of (new_tank_level, water_shortage_flag)
        """
        # Ensure non-negative inputs
        inflow = max(0, inflow)
        outflow = max(0, outflow)
        
        # Calculate new level without constraints
        new_level = self.current_level + inflow - outflow
        
        # Track statistics
        self.total_collected += inflow
        self.total_consumed += outflow
        
        # Check if we have enough water
        water_shortage = False
        if self.current_level + inflow < outflow:
            water_shortage = True
            self.days_insufficient += 1
            # Only provide available water
            actual_outflow = self.current_level + inflow
            new_level = 0
        else:
            actual_outflow = outflow
        
        # Enforce tank capacity constraint
        new_level = max(0, min(new_level, self.capacity))
        
        # Update state
        self.current_level = new_level
        self.history_levels.append(new_level)
        self.history_inflow.append(inflow)
        self.history_outflow.append(actual_outflow if water_shortage else outflow)
        
        return new_level, water_shortage
    
    def get_current_level(self) -> float:
        """Get current tank level in liters."""
        return self.current_level
    
    def get_level_percentage(self) -> float:
        """Get current tank level as percentage of capacity."""
        if self.capacity == 0:
            return 0
        return (self.current_level / self.capacity) * 100
    
    def get_capacity(self) -> float:
        """Get tank capacity in liters."""
        return self.capacity
    
    def set_capacity(self, capacity: float):
        """
        Change tank capacity (useful for simulation variations).
        
        Args:
            capacity: New capacity in liters
        """
        self.capacity = capacity
        # Adjust current level if it exceeds new capacity
        self.current_level = min(self.current_level, capacity)
    
    def can_supply(self, demand: float) -> bool:
        """
        Check if tank can satisfy the demand.
        
        Args:
            demand: Water demand in liters
            
        Returns:
            True if tank has enough water, False otherwise
        """
        return (self.current_level + demand) >= demand or self.current_level >= demand
    
    def supply(self, amount: float) -> Tuple[float, bool]:
        """
        Attempt to supply water from tank.
        
        Args:
            amount: Amount requested (liters)
            
        Returns:
            Tuple of (amount_supplied, insufficient_flag)
        """
        if self.current_level >= amount:
            self.current_level -= amount
            return amount, False
        else:
            supplied = self.current_level
            self.current_level = 0
            return supplied, True
    
    def receive_water(self, amount: float):
        """
        Add water to tank (from rainfall collection).
        
        Args:
            amount: Amount to add (liters)
        """
        self.current_level = min(self.current_level + amount, self.capacity)
    
    def get_history(self) -> dict:
        """
        Get historical data of tank operations.
        
        Returns:
            Dictionary with history arrays
        """
        return {
            'levels': np.array(self.history_levels),
            'inflow': np.array(self.history_inflow),
            'outflow': np.array(self.history_outflow),
            'total_collected': self.total_collected,
            'total_consumed': self.total_consumed,
            'days_insufficient': self.days_insufficient
        }
    
    def get_statistics(self) -> dict:
        """
        Calculate tank statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        levels = np.array(self.history_levels)
        inflow = np.array(self.history_inflow)
        outflow = np.array(self.history_outflow)
        
        stats = {
            'max_level': float(np.max(levels)),
            'min_level': float(np.min(levels)),
            'avg_level': float(np.mean(levels)),
            'avg_fill_percentage': float((np.mean(levels) / self.capacity) * 100) if self.capacity > 0 else 0,
            'total_inflow': float(np.sum(inflow)),
            'total_outflow': float(np.sum(outflow)),
            'total_collected': self.total_collected,
            'total_consumed': self.total_consumed,
            'days_insufficient': self.days_insufficient,
            'efficiency': float((np.sum(outflow) / np.sum(inflow)) * 100) if np.sum(inflow) > 0 else 0
        }
        
        return stats
    
    def reset(self):
        """Reset tank to initial state."""
        self.current_level = self.capacity * config.TANK_INITIAL_LEVEL
        self.history_levels = [self.current_level]
        self.history_inflow = []
        self.history_outflow = []
        self.total_collected = 0
        self.total_consumed = 0
        self.days_insufficient = 0
