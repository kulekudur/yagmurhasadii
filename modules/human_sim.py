"""
Human Simulation Module (Agent-Based Modeling)
Represents workers as agents with time-dependent water consumption
"""

import numpy as np
from typing import List, Tuple, Dict
import config


class Worker:
    """
    Represents a single worker/person in the facility.
    
    Attributes:
    - Active only during working hours (configurable)
    - Consumes water proportional to working duration
    """
    
    def __init__(self, worker_id: int, consumption_per_hour: float = config.CONSUMPTION_PER_WORKER_PER_HOUR):
        """
        Initialize a worker.
        
        Args:
            worker_id: Unique identifier for the worker
            consumption_per_hour: Water consumption per hour in liters
        """
        self.worker_id = worker_id
        self.consumption_per_hour = consumption_per_hour
        self.is_working = False
        self.consumption_today = 0
        
    def set_working(self, is_working: bool):
        """Set whether worker is currently working."""
        self.is_working = is_working
    
    def get_hourly_consumption(self) -> float:
        """
        Get water consumption for the current hour.
        
        Returns:
            Consumption in liters (0 if not working)
        """
        if self.is_working:
            return self.consumption_per_hour
        else:
            return 0
    
    def reset_daily(self):
        """Reset daily consumption counter."""
        self.consumption_today = 0


class WorkforceSimulator:
    """
    Manages a workforce of agents with time-dependent activity patterns.
    
    Features:
    - Variable number of workers
    - Working hours scheduling (e.g., 9AM-5PM)
    - Realistic consumption patterns
    """
    
    def __init__(
        self,
        worker_count: int = config.WORKER_COUNT_DEFAULT,
        consumption_per_hour: float = config.CONSUMPTION_PER_WORKER_PER_HOUR,
        work_start_hour: int = config.WORK_START_HOUR,
        work_end_hour: int = config.WORK_END_HOUR
    ):
        """
        Initialize the workforce simulator.
        
        Args:
            worker_count: Number of workers
            consumption_per_hour: Consumption per worker per hour (liters)
            work_start_hour: Hour work starts (0-23)
            work_end_hour: Hour work ends (0-23)
        """
        self.workers = [Worker(i, consumption_per_hour) for i in range(worker_count)]
        self.worker_count = worker_count
        self.work_start_hour = work_start_hour
        self.work_end_hour = work_end_hour
        self.consumption_per_hour = consumption_per_hour
        self.history_hourly_consumption = []
        self.history_daily_consumption = []
        
    def is_working_hour(self, hour: int) -> bool:
        """
        Check if given hour is within working hours.
        
        Args:
            hour: Hour (0-23)
            
        Returns:
            True if hour is within working hours
        """
        if self.work_start_hour < self.work_end_hour:
            # Normal case: 9AM to 5PM
            return self.work_start_hour <= hour < self.work_end_hour
        else:
            # Wrap-around case: 9PM to 5AM (night shift)
            return hour >= self.work_start_hour or hour < self.work_end_hour
    
    def update_hour(self, hour: int):
        """
        Update workforce activity for the given hour.
        
        Args:
            hour: Hour of day (0-23)
        """
        working = self.is_working_hour(hour)
        for worker in self.workers:
            worker.set_working(working)
    
    def get_hourly_consumption(self, hour: int) -> float:
        """
        Calculate total water consumption for a specific hour.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Total consumption in liters
        """
        self.update_hour(hour)
        total = sum(worker.get_hourly_consumption() for worker in self.workers)
        self.history_hourly_consumption.append(total)
        return total
    
    def get_daily_consumption(self) -> float:
        """
        Calculate total daily consumption (sum of all hours).
        
        Returns:
            Daily consumption in liters
        """
        daily_total = sum(self.history_hourly_consumption)
        self.history_daily_consumption.append(daily_total)
        return daily_total
    
    def set_worker_count(self, count: int):
        """
        Change the number of workers.
        
        Args:
            count: New number of workers
        """
        if count > len(self.workers):
            # Add more workers
            for i in range(len(self.workers), count):
                self.workers.append(Worker(i, self.consumption_per_hour))
        elif count < len(self.workers):
            # Remove workers
            self.workers = self.workers[:count]
        
        self.worker_count = count
    
    def get_worker_count(self) -> int:
        """Get current number of workers."""
        return self.worker_count
    
    def reset_daily(self):
        """Reset daily counters and history for new day."""
        self.history_hourly_consumption = []
        for worker in self.workers:
            worker.reset_daily()
    
    def get_statistics(self) -> dict:
        """
        Calculate consumption statistics.
        
        Returns:
            Dictionary with consumption metrics
        """
        if not self.history_daily_consumption:
            return {
                'total_consumption': 0,
                'avg_daily': 0,
                'max_hourly': 0,
                'min_hourly': 0,
                'working_hours_per_day': self._get_working_hours_count()
            }
        
        daily_array = np.array(self.history_daily_consumption)
        hourly_array = np.array(self.history_hourly_consumption)
        
        stats = {
            'total_consumption': float(np.sum(daily_array)),
            'avg_daily': float(np.mean(daily_array)),
            'max_daily': float(np.max(daily_array)),
            'min_daily': float(np.min(daily_array)),
            'max_hourly': float(np.max(hourly_array)) if len(hourly_array) > 0 else 0,
            'min_hourly': float(np.min(hourly_array)) if len(hourly_array) > 0 else 0,
            'working_hours_per_day': self._get_working_hours_count(),
            'per_worker_daily': float(np.mean(daily_array) / self.worker_count) if self.worker_count > 0 else 0
        }
        
        return stats
    
    def _get_working_hours_count(self) -> int:
        """Calculate number of working hours per day."""
        if self.work_start_hour < self.work_end_hour:
            return self.work_end_hour - self.work_start_hour
        else:
            return (24 - self.work_start_hour) + self.work_end_hour
    
    def get_peak_consumption_hour(self) -> Tuple[int, float]:
        """
        Find the hour with peak consumption.
        
        Returns:
            Tuple of (hour, peak_consumption)
        """
        if not self.history_hourly_consumption:
            return 0, 0
        
        peak_value = max(self.history_hourly_consumption)
        peak_hour = self.history_hourly_consumption.index(peak_value)
        return peak_hour, peak_value
    
    def get_consumption_profile(self) -> List[Tuple[int, float]]:
        """
        Get consumption profile by hour (showing typical consumption per hour).
        
        Returns:
            List of (hour, typical_consumption) tuples
        """
        profile = []
        for hour in range(24):
            if self.is_working_hour(hour):
                consumption = self.worker_count * self.consumption_per_hour
            else:
                consumption = 0
            profile.append((hour, consumption))
        
        return profile
