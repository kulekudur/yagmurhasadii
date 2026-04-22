"""
Example Usage Scripts for Rainwater Harvesting Simulation Platform
Demonstrates how to use the modules directly in Python code
"""

import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.simulation_engine import SimulationEngine
from modules.rain_sim import RainfallSimulator
from modules.tank_sim import StorageTank
from modules.human_sim import WorkforceSimulator
from modules.economy import EconomicAnalyzer
import config


def example_1_basic_simulation():
    """
    Example 1: Run a basic simulation with default parameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Simulation with Default Parameters")
    print("="*60)
    
    # Create simulation engine
    engine = SimulationEngine()
    
    # Run full year simulation
    results = engine.run_full_simulation()
    
    # Display results
    print("\n📊 Simulation Results:")
    print(f"   Total Water Collected: {results['water_metrics']['total_collected']:,.0f} L")
    print(f"   Total Water Consumed:  {results['water_metrics']['total_consumed']:,.0f} L")
    print(f"   Utilization Rate:      {results['water_metrics']['utilization_rate']:.1f}%")
    print(f"   Shortage Days:         {results['water_metrics']['shortage_days']} days")
    
    print("\n💰 Economic Analysis:")
    eco = results['economic_metrics']['financial']
    print(f"   Water Savings:         ₺ {eco['cost_saved']:,.0f}")
    print(f"   Total Investment:      ₺ {eco['total_investment']:,.0f}")
    print(f"   Net Benefit:           ₺ {eco['net_benefit']:,.0f}")
    print(f"   ROI:                   {eco['roi_percentage']:.1f}%")
    print(f"   Payback Period:        {eco['payback_years']:.2f} years")

    return results


def example_2_custom_parameters():
    """
    Example 2: Run simulation with custom parameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    
    # Create engine with custom parameters
    engine = SimulationEngine(
        roof_area=1000,          # 1000 m² roof (double default)
        roof_efficiency=0.90,    # 90% efficiency
        tank_capacity=100000,    # 100,000 liters (double)
        worker_count=100,        # 100 workers (double)
        rain_seed=123            # Custom seed for reproducibility
    )
    
    print("\n🔧 Custom Parameters:")
    print(f"   Roof Area:       {engine.roof_area} m²")
    print(f"   Efficiency:      {engine.roof_efficiency:.0%}")
    print(f"   Tank Capacity:   {engine.tank.get_capacity():,.0f} L")
    print(f"   Worker Count:    {engine.workforce.get_worker_count()}")
    
    # Run simulation
    results = engine.run_full_simulation()
    
    # Compare results
    print("\n📈 Results with Custom Parameters:")
    print(f"   Total Collected: {results['water_metrics']['total_collected']:,.0f} L")
    print(f"   Total Consumed:  {results['water_metrics']['total_consumed']:,.0f} L")
    eco = results['economic_metrics']['financial']
    print(f"   ROI:             {eco['roi_percentage']:.1f}%")

    return results


def example_3_rainfall_analysis():
    """
    Example 3: Detailed rainfall analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Rainfall Analysis")
    print("="*60)
    
    # Create rainfall simulator
    rain_sim = RainfallSimulator(seed=42)
    
    # Generate annual rainfall
    rainfall_data = rain_sim.generate_annual_rainfall()
    
    # Get statistics
    stats = rain_sim.get_statistics()
    
    print("\n🌧️  Rainfall Statistics:")
    print(f"   Total Rainfall:      {stats['total_rainfall']:.1f} mm")
    print(f"   Rainy Days:          {stats['rainy_days']} days")
    print(f"   Rain Probability:    {stats['rain_probability_actual']:.1%}")
    print(f"   Average Daily:       {stats['average_daily']:.2f} mm")
    print(f"   Max Daily:           {stats['max_daily']:.1f} mm")
    print(f"   Min Daily:           {stats['min_daily']:.1f} mm")
    print(f"   Std Deviation:       {stats['std_dev']:.2f} mm")
    
    # Show monthly summary
    print("\n📅 Monthly Rainfall Summary:")
    monthly = rain_sim.get_monthly_summary()
    print(monthly)
    
    return stats, rainfall_data


def example_4_tank_dynamics():
    """
    Example 4: Tank system dynamics
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Tank Dynamics")
    print("="*60)
    
    # Create tank
    tank = StorageTank(capacity=50000)
    
    print("\n🏭 Tank Configuration:")
    print(f"   Capacity:        {tank.get_capacity():,.0f} L")
    print(f"   Initial Level:   {tank.get_current_level():,.0f} L")
    print(f"   Initial %:       {tank.get_level_percentage():.1f}%")
    
    # Simulate some days
    print("\n📊 Daily Simulation (10 days):")
    print(f"{'Day':<5} {'Inflow':<10} {'Outflow':<10} {'Level':<12} {'%':<6} {'Shortage'}")
    print("-" * 60)
    
    for day in range(10):
        inflow = 1000 + (day * 100)      # Increasing inflow
        outflow = 1500 - (day * 50)      # Decreasing outflow
        
        level, shortage = tank.update(inflow, outflow)
        
        print(f"{day+1:<5} {inflow:<10.0f} {outflow:<10.0f} {level:<12,.0f} {tank.get_level_percentage():<6.1f} {'Yes' if shortage else 'No'}")
    
    # Get statistics
    stats = tank.get_statistics()
    print("\n📈 Tank Statistics:")
    print(f"   Max Level:       {stats['max_level']:,.0f} L")
    print(f"   Min Level:       {stats['min_level']:,.0f} L")
    print(f"   Avg Level:       {stats['avg_level']:,.0f} L")
    print(f"   Efficiency:      {stats['efficiency']:.1f}%")

    return tank, stats


def example_5_economic_analysis():
    """
    Example 5: Economic analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Economic Analysis")
    print("="*60)
    
    # Create analyzer
    analyzer = EconomicAnalyzer(
        water_price=0.50,
        tank_cost=5000,
        maintenance_cost_annual=500,
        installation_cost=2000
    )
    
    # Simulate annual water collection
    annual_collected = 200000  # liters
    annual_consumed = 160000   # liters (80% utilization)
    
    print("\n💧 Water Metrics:")
    print(f"   Annual Collected:  {annual_collected:,.0f} L")
    print(f"   Annual Consumed:   {annual_consumed:,.0f} L")
    print(f"   Utilization:       {(annual_consumed/annual_collected)*100:.1f}%")
    
    # Calculate savings
    savings = analyzer.calculate_water_savings(annual_collected, annual_consumed)
    print("\n💰 Water Savings:")
    print(f"   Cost Saved:        ₺ {savings['cost_saved']:,.0f}")
    print(f"   Water Saved:       {savings['water_saved_liters']:,.0f} L")
    
    # Calculate system costs
    costs = analyzer.calculate_system_costs(years=1)
    print("\n🏗️  System Costs (Year 1):")
    print(f"   Installation:      ₺ {costs['installation_cost']:,.0f}")
    print(f"   Tank Cost:         ₺ {costs['tank_cost']:,.0f}")
    print(f"   Maintenance:       ₺ {costs['maintenance_annual']:,.0f}")
    print(f"   Total Cost:        ₺ {costs['total_cost']:,.0f}")
    
    # Calculate ROI
    roi = analyzer.calculate_roi(annual_collected, annual_consumed, years=1)
    print("\n📊 Return on Investment (1-Year):")
    print(f"   ROI %:             {roi['roi_percentage']:.1f}%")
    print(f"   Net Benefit:       ₺ {roi['net_benefit']:,.0f}")
    print(f"   Payback Period:    {roi['payback_period_years']:.2f} years")
    print(f"   Break-Even:        {'Yes ✓' if roi['break_even'] else 'No ✗'}")
    
    # Breakeven analysis
    breakeven = analyzer.get_breakeven_analysis(annual_collected)
    print("\n🎯 Break-Even Analysis:")
    print(f"   Annual Water Value:  ₺ {breakeven['annual_water_value']:,.0f}")
    print(f"   Years to Break-Even: {breakeven['years_to_breakeven']:.2f} years")
    print(f"   Economically Viable: {'Yes ✓' if breakeven['economically_viable'] else 'No ✗'}")

    return analyzer, roi


def example_6_worker_simulation():
    """
    Example 6: Worker/agent simulation
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Worker Simulation")
    print("="*60)
    
    # Create workforce
    workforce = WorkforceSimulator(
        worker_count=50,
        consumption_per_hour=2.0,
        work_start_hour=9,
        work_end_hour=17
    )
    
    print(f"\n👥 Workforce Configuration:")
    print(f"   Worker Count:        {workforce.get_worker_count()}")
    print(f"   Working Hours:       {workforce.work_start_hour}:00 - {workforce.work_end_hour}:00")
    print(f"   Consumption/hour:    {workforce.consumption_per_hour} L/worker")
    
    # Get consumption profile for a day
    print(f"\n📊 Hourly Consumption Profile (24-hour):")
    print(f"{'Hour':<6} {'Working':<10} {'Consumption (L)':<20}")
    print("-" * 40)
    
    total_daily = 0
    for hour in range(24):
        is_working = workforce.is_working_hour(hour)
        hourly_consumption = workforce.get_hourly_consumption(hour)
        total_daily += hourly_consumption
        
        status = "Yes ✓" if is_working else "No"
        print(f"{hour:<6} {status:<10} {hourly_consumption:<20,.0f}")
    
    print(f"\n💡 Daily Total Consumption: {total_daily:,.0f} L")
    print(f"   Peak Hour: {workforce.work_start_hour}:00-{workforce.work_start_hour+1}:00")

    return workforce


def run_all_examples():
    """
    Run all examples in sequence
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  RAINWATER HARVESTING SIMULATION - EXAMPLES ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Run each example
        example_1_basic_simulation()
        example_2_custom_parameters()
        example_3_rainfall_analysis()
        example_4_tank_dynamics()
        example_5_economic_analysis()
        example_6_worker_simulation()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Start the web app:  streamlit run app.py")
        print("  2. Review the code in modules/")
        print("  3. Modify config.py for different scenarios")
        print("  4. Create custom simulations using the examples above")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rainwater Harvesting Simulation Examples")
    parser.add_argument("--example", type=int, default=0, 
                       help="Run specific example (1-6) or 0 for all")
    
    args = parser.parse_args()
    
    if args.example == 0:
        run_all_examples()
    elif args.example == 1:
        example_1_basic_simulation()
    elif args.example == 2:
        example_2_custom_parameters()
    elif args.example == 3:
        example_3_rainfall_analysis()
    elif args.example == 4:
        example_4_tank_dynamics()
    elif args.example == 5:
        example_5_economic_analysis()
    elif args.example == 6:
        example_6_worker_simulation()
    else:
        print(f"Unknown example: {args.example}")
        print("Use --example with a number 0-6")
