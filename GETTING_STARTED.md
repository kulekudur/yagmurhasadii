# Getting Started Guide

## Installation

### Step 1: Navigate to Project Directory
```bash
cd "c:\Users\Ali Rıza AKBAY\Desktop\modelleme"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **Streamlit** - Web UI framework
- **Plotly** - 3D visualization
- **NumPy** - Numerical computing
- **Pandas** - Data analysis
- **SciPy** - Scientific computing

---

## Running the Application

### Option 1: Web Application (Recommended)

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`

### Option 2: Run Examples (Python Scripts)

```bash
# Run all examples
python examples.py

# Run specific example (1-6)
python examples.py --example 1
```

### Option 3: Quick Test

```bash
python setup.bat  # Windows
# or
./setup.sh        # macOS/Linux
```

---

## First-Time Usage

### 1. Default Simulation
1. Open the app: `streamlit run app.py`
2. Click **▶️ Run Simulation** in the sidebar
3. Wait ~2-5 seconds for simulation to complete
4. Explore the tabs:
   - **📊 Overview**: Summary statistics
   - **🎬 3D Visualization**: Interactive 3D scene with time controls
   - **📈 Graphs**: Time series analysis
   - **💰 Economics**: Financial analysis
   - **📥 Export**: Download data

### 2. Adjust Parameters
Use the sliders in the sidebar to modify:
- **Roof Area**: Building collection area (100-2000 m²)
- **Collection Efficiency**: System efficiency (0-100%)
- **Tank Capacity**: Storage size (5K-500K liters)
- **Worker Count**: People consuming water (1-300)
- **Water Price**: Cost per liter for economic analysis

### 3. Run Multiple Simulations
- Change any parameters
- Click **▶️ Run Simulation** again
- Compare results across different configurations

---

## Understanding the Results

### Water Metrics
- **Total Collected**: Water captured from rainfall
- **Total Consumed**: Water actually used
- **Shortage Days**: Days when demand exceeded supply
- **Utilization Rate**: Efficiency of water use (collected vs consumed)

### Economic Results
- **ROI %**: Return on investment percentage
- **Payback Period**: Years to recover the investment
- **Cost Saved**: Money saved by using harvested water
- **Net Benefit**: Total savings minus system cost

### Example Results (Default Configuration)
```
Annual Rainfall:        ~250 mm
Total Collected:        ~106,000 liters
Total Consumed:         ~114,000 liters
Shortage Days:          3-5 days
ROI (Year 1):           -85% to -70% (system pays off in 1-2 years)
```

---

## Project Structure

```
modelleme/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration parameters
├── requirements.txt                # Python dependencies
├── examples.py                     # Example scripts
├── README.md                       # Full documentation
├── GETTING_STARTED.md             # This file
│
├── modules/
│   ├── rain_sim.py                # Rainfall generation
│   ├── tank_sim.py                # Tank dynamics
│   ├── human_sim.py               # Worker simulation
│   ├── economy.py                 # Economic analysis
│   ├── visualization.py           # 3D visualization
│   └── simulation_engine.py       # Main engine
│
└── assets/                        # (For future assets)
```

---

## Common Tasks

### Task 1: Change Simulation Scenario
1. Open app, adjust sliders for your scenario
2. Click **▶️ Run Simulation**
3. View results in the tabs

Example scenarios:
- **Large Building**: Roof 2000 m², Tank 100K L, 100 workers
- **Small Office**: Roof 300 m², Tank 20K L, 30 workers
- **Factory**: Roof 5000 m², Tank 200K L, 300 workers

### Task 2: Export Results
1. Go to **📥 Export** tab
2. View daily data table
3. Click **📥 Download Daily Data (CSV)** to export
4. Download **Summary Report** as TXT file

### Task 3: Understand 3D Visualization
- Use **Day** slider to browse through the year
- Use **Hour** slider to see time-of-day effects
- Watch the tank fill level change based on rainfall/consumption
- See rain particles when rainfall occurs

### Task 4: Compare Scenarios
1. Run Simulation A with parameters X
2. Note the ROI and savings
3. Modify parameters for Scenario B
4. Run Simulation B
5. Compare results side-by-side

---

## Troubleshooting

### Issue: "streamlit: command not found"
**Solution:**
```bash
pip install streamlit==1.28.1
```

### Issue: "ModuleNotFoundError: No module named 'plotly'"
**Solution:**
```bash
pip install plotly==5.18.0
```

### Issue: App loads but 3D scene is blank
**Solution:**
- Try refreshing the page
- Check your internet connection (Plotly uses CDN)
- Try a different browser

### Issue: Simulation is very slow
**Solution:**
- Reduce worker count temporarily (for testing)
- Close other applications
- Try with fewer workers initially

### Issue: Python not found
**Solution:**
- Install Python from https://www.python.org
- Add Python to PATH
- Use `python3` instead of `python` on macOS

---

## Configuration Customization

To permanently change default values, edit `config.py`:

```python
# Change default roof area
ROOF_AREA_DEFAULT = 750  # was 500

# Change working hours
WORK_START_HOUR = 8   # was 9
WORK_END_HOUR = 18    # was 17

# Change water price
WATER_PRICE = 0.75    # was 0.50
```

Then restart the app: `streamlit run app.py`

---

## Next Steps

1. **Run Examples**: `python examples.py` to see how the system works
2. **Explore Code**: Review modules to understand implementation
3. **Modify Parameters**: Experiment with different scenarios
4. **Extend System**: Add new modules (solar, wind, etc.)
5. **Export Results**: Generate reports for analysis

---

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2 GB minimum (4 GB recommended)
- **Disk Space**: ~200 MB
- **Browser**: Chrome, Firefox, Safari, or Edge (for Streamlit UI)

---

## Tips & Tricks

### Tip 1: Reproducible Results
A fixed **Rainfall Seed** produces the same rainfall pattern each time. Use this for consistent testing.

### Tip 2: Economic Break-Even
Look for the year when "Cost Saved" exceeds "Total Investment" in the economics tab.

### Tip 3: Water Shortage Optimization
- Increase tank capacity to handle dry periods
- Increase roof area to collect more water
- Reduce consumption or worker count

### Tip 4: Performance Comparison
Save results from multiple runs to compare:
- Which parameter has most impact?
- What's the optimal tank size?
- Best ROI configuration?

---

## Learning Path

### Beginner (First Time)
1. Run default simulation
2. Explore the 3D visualization
3. Review the Overview tab
4. Check the Economics tab

### Intermediate (Second Time)
1. Run examples.py to see code usage
2. Modify a parameter and re-run
3. Compare results
4. Export and analyze data

### Advanced (Custom Use)
1. Review module source code
2. Modify config.py for your scenario
3. Create custom parameter combinations
4. Extend with new modules

---

## Additional Resources

- **Main README**: `README.md` - Complete documentation
- **Source Code**: All modules in `modules/` folder
- **Examples**: `examples.py` - Example usage patterns
- **Configuration**: `config.py` - All system parameters

---

## Support & Questions

### Common Questions

**Q: How do I interpret the "Shortage Days"?**
A: Shortage days indicate when water demand exceeded collected supply. More days = need larger tank or higher collection area.

**Q: What does ROI of -50% mean?**
A: Negative ROI in Year 1 means the system costs more than it saves. It will reach break-even in ~2 years and then become profitable.

**Q: Can I use real rainfall data?**
A: Currently using stochastic generation. Real data integration is planned for v2.0.

**Q: How do I add new modules?**
A: Create a new file in `modules/` and integrate into `simulation_engine.py` and `app.py`.

---

**Version**: 1.0  
**Last Updated**: April 2026  
**Status**: Ready to Use ✓
