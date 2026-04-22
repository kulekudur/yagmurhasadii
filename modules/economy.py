"""
Economic Analysis Module
Calculates costs, savings, and ROI for the rainwater harvesting system.

Ekonomik analiz, basit "toplam maliyet / yıllık tasarruf" yerine
iskonto edilmiş nakit akışları (DCF) üzerinden hesaplanır:

- İlk yatırım  I = installation_cost + tank_cost (yıl 0).
- Yıllık net nakit akışı CF_t = S_t - M   (S_t reel artışlı tasarruf, M yıllık bakım)
- İskonto edilmiş nakit akışı DCF_t = CF_t / (1+r)^t
- Discounted payback: en küçük n için Σ_{t=1..n} DCF_t >= I; fraksiyonel yıl döner.
- Lifespan içinde ulaşılmazsa payback = None.
"""

from typing import Dict, List, Optional

import numpy as np

import config


class EconomicAnalyzer:
    """
    Analyzes the economic viability of a rainwater harvesting system.

    Metrics:
    - Water savings (liters and cost)
    - System costs (installation, maintenance)
    - Net Present Value (NPV)
    - Discounted payback period (iskonto edilmiş geri ödeme)
    """

    def __init__(
        self,
        water_price: float = config.WATER_PRICE,
        tank_cost: float = config.TANK_COST,
        maintenance_cost_annual: float = config.MAINTENANCE_COST,
        installation_cost: float = config.INSTALLATION_COST,
        water_price_escalation: float = config.WATER_PRICE_ESCALATION,
        discount_rate: float = config.DISCOUNT_RATE,
        system_lifespan_years: int = config.SYSTEM_LIFESPAN_YEARS,
    ):
        """
        Initialize economic analyzer.

        Args:
            water_price: Price per liter of water (₺)
            tank_cost: Cost of storage tank (₺)
            maintenance_cost_annual: Annual maintenance cost (₺)
            installation_cost: Installation/setup cost (₺)
            water_price_escalation: Reel yıllık su fiyatı artışı (e.g. 0.03 = %3)
            discount_rate: Reel iskonto oranı (e.g. 0.05 = %5)
            system_lifespan_years: Ekonomik ömür (yıl)
        """
        self.water_price = water_price
        self.tank_cost = tank_cost
        self.maintenance_cost_annual = maintenance_cost_annual
        self.installation_cost = installation_cost
        self.water_price_escalation = water_price_escalation
        self.discount_rate = discount_rate
        self.system_lifespan_years = int(system_lifespan_years)
        
    def calculate_water_savings(
        self,
        water_collected: float,
        water_consumed_from_tank: float
    ) -> Dict[str, float]:
        """
        Calculate water savings from rainwater harvesting.
        
        Args:
            water_collected: Total water collected (liters)
            water_consumed_from_tank: Water actually used from tank (liters)
            
        Returns:
            Dictionary with savings metrics
        """
        savings_liters = water_collected
        savings_cost = savings_liters/1000 * self.water_price 
        
        if water_collected > 0:
            utilization_rate = (water_consumed_from_tank / water_collected) * 100
        else:
            utilization_rate = 0
        
        return {
            'water_saved_liters': savings_liters,
            'cost_saved': savings_cost,
            'utilization_rate': utilization_rate,
            'collected_liters': water_collected
        }
    
    def calculate_system_costs(self, years: int = 1) -> Dict[str, float]:
        """
        Calculate total system costs.
        
        Args:
            years: Number of years to calculate for
            
        Returns:
            Dictionary with cost breakdown
        """
        total_cost = self.installation_cost + self.tank_cost
        maintenance_total = self.maintenance_cost_annual * years
        total_cost += maintenance_total
        
        return {
            'installation_cost': self.installation_cost,
            'tank_cost': self.tank_cost,
            'maintenance_annual': self.maintenance_cost_annual,
            'maintenance_total': maintenance_total,
            'total_cost': total_cost,
            'cost_per_year': total_cost / years if years > 0 else 0
        }
    
    def calculate_discounted_payback(
        self,
        water_savings_y1: float,
        maintenance_annual: float,
        initial_investment: float,
    ) -> Dict:
        """
        Compute the discounted payback period and the underlying cash flow table.

        Model (reel terimler):
            S_t   = water_savings_y1 * (1 + e)^(t-1)
            CF_t  = S_t - maintenance_annual
            DCF_t = CF_t / (1 + r)^t
            payback = min n  s.t. Σ_{t=1..n} DCF_t ≥ initial_investment

        Args:
            water_savings_y1: Yıl-1 reel kaba su tasarrufu (TL)
            maintenance_annual: Yıllık reel bakım gideri (TL)
            initial_investment: Yıl-0 tek seferlik yatırım (TL)

        Returns:
            Dict:
                payback_years_discounted (Optional[float]): fraksiyonel yıl ya da None
                payback_years_simple     (Optional[float]): iskontosuz/eskalasyonsuz
                npv (float): lifespan sonunda net bugünkü değer
                cash_flow_table (list[dict]): yıl bazlı satırlar
                assumptions (dict)
        """
        e = float(self.water_price_escalation)
        r = float(self.discount_rate)
        life = int(self.system_lifespan_years)

        cum = 0.0
        rows: List[Dict] = []
        payback: Optional[float] = None
        for t in range(1, life + 1):
            s_t = water_savings_y1 * ((1.0 + e) ** (t - 1))
            cf_t = s_t - maintenance_annual
            dcf_t = cf_t / ((1.0 + r) ** t)
            prev = cum
            cum += dcf_t
            rows.append({
                "year": t,
                "savings": s_t,
                "maintenance": maintenance_annual,
                "net_cf": cf_t,
                "dcf": dcf_t,
                "cum_dcf": cum,
            })
            # İlk kez yatırımı aşan kümülatif DCF → fraksiyonel yıl
            if payback is None and cum >= initial_investment and dcf_t > 0:
                payback = (t - 1) + (initial_investment - prev) / dcf_t

        net_y1 = water_savings_y1 - maintenance_annual
        if net_y1 > 0:
            simple_payback: Optional[float] = initial_investment / net_y1
        else:
            simple_payback = None

        return {
            "payback_years_discounted": payback,
            "payback_years_simple": simple_payback,
            "npv": cum - initial_investment,
            "cash_flow_table": rows,
            "assumptions": {
                "water_price_escalation": e,
                "discount_rate": r,
                "system_lifespan_years": life,
                "initial_investment": initial_investment,
                "maintenance_annual": maintenance_annual,
                "water_savings_y1": water_savings_y1,
            },
        }

    def calculate_roi(
        self,
        water_collected: float,
        water_consumed_from_tank: float,
        years: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate Return on Investment (ROI) using simple model.

        Net Benefit = Total Savings - Total Costs over lifespan
        ROI% = (Net Benefit / Initial Investment) * 100

        Args:
            water_collected: Total water collected in the simulated year (liters)
            water_consumed_from_tank: Water used from tank in the simulated year (L)
            years: Simülasyon süresi (yıl). 1 kabul edilir; ömür içi projeksiyon
                   `system_lifespan_years`'a göre yapılır.

        Returns:
            Dictionary with ROI metrics + payback details.
        """
        savings = self.calculate_water_savings(water_collected, water_consumed_from_tank)
        annual_savings = savings["cost_saved"] / max(1, years)
        initial_investment = float(self.installation_cost + self.tank_cost)
        total_maintenance = float(self.maintenance_cost_annual) * self.system_lifespan_years
        total_costs = initial_investment + total_maintenance
        total_savings = annual_savings * self.system_lifespan_years
        net_benefit = total_savings - total_costs
        
        if initial_investment > 0:
            roi_percentage = (net_benefit / initial_investment) * 100.0
        else:
            roi_percentage = 0.0

        # Simple payback
        net_annual = annual_savings - self.maintenance_cost_annual
        if net_annual > 0:
            payback_simple = initial_investment / net_annual
        else:
            payback_simple = None

        return {
            "roi_percentage": roi_percentage,
            "net_benefit": net_benefit,  # toplam net fayda
            "npv": net_benefit,  # geriye uyum
            "payback_years_simple": payback_simple,
            "payback_years_discounted": payback_simple,  # basit modelde aynı
            "payback_period_years": payback_simple,
            "annual_savings": annual_savings,
            "total_savings": total_savings,
            "initial_investment": initial_investment,
            "installation_cost": float(self.installation_cost),
            "tank_cost": float(self.tank_cost),
            "maintenance_annual": float(self.maintenance_cost_annual),
            "total_investment": total_costs,  # toplam maliyet
            "break_even": payback_simple is not None and payback_simple <= self.system_lifespan_years,
            "cash_flow_table": [],  # basit modelde boş
            "assumptions": {
                "water_price_escalation": self.water_price_escalation,
                "discount_rate": self.discount_rate,
                "system_lifespan_years": self.system_lifespan_years,
                "initial_investment": initial_investment,
                "maintenance_annual": self.maintenance_cost_annual,
                "water_savings_y1": annual_savings,
            },
        }
    
    def get_breakeven_analysis(
        self,
        annual_water_collection: float
    ) -> Dict[str, float]:
        """
        Analyze breakeven point (when savings equal costs).
        
        Args:
            annual_water_collection: Expected annual water collection (liters)
            
        Returns:
            Dictionary with breakeven metrics
        """
        annual_value = annual_water_collection * self.water_price
        costs = self.calculate_system_costs(years=1)
        
        breakeven_cost = costs['total_cost']
        if annual_value > 0:
            years_to_breakeven = breakeven_cost / annual_value
        else:
            years_to_breakeven = float('inf')
        
        return {
            'annual_water_value': annual_value,
            'system_cost': breakeven_cost,
            'years_to_breakeven': years_to_breakeven,
            'economically_viable': years_to_breakeven < 20  # Typical system lifespan
        }
    
    def sensitivity_analysis(
        self,
        annual_water_collection: float,
        price_range: tuple = (0.3, 0.7)
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on water price.
        
        Args:
            annual_water_collection: Expected annual collection (liters)
            price_range: Tuple of (min_price, max_price)
            
        Returns:
            Dictionary with ROI at different price points
        """
        scenarios = {}
        original_price = self.water_price
        
        price_points = np.linspace(price_range[0], price_range[1], 5)
        
        for price in price_points:
            self.water_price = price
            roi = self.calculate_roi(
                water_collected=annual_water_collection,
                water_consumed_from_tank=annual_water_collection * 0.8,  # Assume 80% utilization
                years=1
            )
            scenarios[f'{price:.2f}_per_liter'] = roi
        
        self.water_price = original_price  # Restore original price
        return scenarios
    
    def get_annual_summary(
        self,
        water_collected: float,
        water_consumed: float,
        worker_count: int
    ) -> Dict:
        """
        Generate comprehensive annual economic summary.
        
        Args:
            water_collected: Annual water collected (liters)
            water_consumed: Annual water consumed from tank (liters)
            worker_count: Number of workers
            
        Returns:
            Comprehensive summary dictionary
        """
        savings = self.calculate_water_savings(water_collected, water_consumed)
        costs = self.calculate_system_costs(years=1)
        roi = self.calculate_roi(water_collected, water_consumed, years=1)
        breakeven = self.get_breakeven_analysis(water_collected)
        
        return {
            'overview': {
                'year': 2024,
                'worker_count': worker_count,
            },
            'water_metrics': {
                'collected_liters': savings['collected_liters'],
                'consumed_liters': savings['water_saved_liters'],
                'utilization_rate': savings['utilization_rate']
            },
            'financial': {
                'cost_saved': savings['cost_saved'],
                'total_investment': roi['total_investment'],
                'initial_investment': roi['initial_investment'],
                'installation_cost': roi['installation_cost'],
                'tank_cost': roi['tank_cost'],
                'maintenance_annual': roi['maintenance_annual'],
                'net_benefit': roi['net_benefit'],
                'npv': roi['npv'],
                'roi_percentage': roi['roi_percentage'],
                'payback_years': roi['payback_years_discounted'],
                'payback_years_simple': roi['payback_years_simple'],
                'payback_years_discounted': roi['payback_years_discounted'],
                'annual_savings': roi['annual_savings'],
                'cash_flow_table': roi['cash_flow_table'],
                'assumptions': roi['assumptions'],
            },
            'breakeven': breakeven
        }
