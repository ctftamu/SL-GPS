# 🚀 Getting Started with SL-GPS Frontend GUI

Welcome! This guide will get you up and running with the SL-GPS graphical interface in **5 minutes**.

## Installation (2 minutes)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ctftamu/SL-GPS.git
cd SL-GPS
```

### Step 2: Install Dependencies
```bash
# Install all requirements (including GUI)
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

**That's it!** You're ready to go.

## Launch the GUI (1 minute)

```bash
python -m frontend
```

Your browser will **automatically open** to `http://localhost:7860`

You should see a beautiful interface with three tabs:
- 📊 Generate Dataset
- 🧠 Train Neural Network
- 📖 Documentation

## First Run: Quick Test (2 minutes)

### Follow these steps to test everything:

**1. Generate Dataset Tab**
- Click **Upload Cantera Mechanism** 
  - For testing, use `src/slgps/gri30.cti` or `src/slgps/nHeptane.cti`
  - Or download from [Cantera](https://cantera.org/databases/mechanisms/)

- Keep default parameters OR for quick test change:
  - **Number of Cases**: 5 (instead of 100) - runs in ~2-5 minutes
  - **Temperature**: 1000-1500 K (instead of 800-2300)

- Click **🚀 Generate Dataset**
- Wait for completion (you'll see green status messages)

**2. Train Neural Network Tab**
- Species should auto-populate
- Keep default architecture
- Click **🚀 Train Neural Network**
- Wait for training completion

**3. Check Results**
- Look for `generated_data/` folder with:
  - `model.h5` - Your trained neural network
  - `scaler.pkl` - Input normalizer
  - `data.csv` - Training data
  - `species.csv` - Species importance masks

## Next Steps

After the quick test:

1. **Read the Documentation**
   - In the GUI: Click the **📖 Documentation** tab
   - Online: https://ctftamu.github.io/SL-GPS/
   - Quick ref: `FRONTEND_QUICKSTART.md`

2. **Run a Real Experiment**
   - Use your own mechanism file
   - Set realistic parameters for your fuel/conditions
   - Generate full dataset (n_cases = 100-500)
   - Train with custom NN architecture if needed

3. **Use the Results**
   - See `SL_GPS.py` to use model in adaptive simulations
   - See `display_sim_data.py` to visualize results

## File You'll Need

Most important files:

| File | Purpose |
|------|---------|
| `frontend/app.py` | The GUI application |
| `FRONTEND_QUICKSTART.md` | Quick reference (parameters, troubleshooting) |
| `docs/frontend.md` | Complete frontend documentation |
| `docs/setup.md` | Installation & troubleshooting |
| `docs/api.md` | Full API reference |

## Common Questions

**Q: What mechanism file should I use?**  
A: Any Cantera CTI file. Examples:
- `gri30.cti` - General combustion mechanism
- `nHeptane.cti` - For heptane combustion
- Download from https://cantera.org/databases/mechanisms/

**Q: How long does data generation take?**  
A: Depends on `n_cases`:
- 5 cases: 2-5 minutes
- 20 cases: 5-15 minutes
- 100 cases: 30-60 minutes
- 500 cases: 2-4 hours

**Q: Can I customize the neural network?**  
A: Yes! See `docs/code_structure.md` and edit `src/slgps/mech_train.py::spec_train()` for custom layers.

**Q: Port 7860 is in use**  
A: Run `python -m frontend --server_port 7861` for a different port.

**Q: How do I use my trained model?**  
A: Load `model.h5` and `scaler.pkl` in `src/slgps/SL_GPS.py` for adaptive simulations.

## Where to Go From Here

- **Stuck?** → `FRONTEND_QUICKSTART.md` (troubleshooting section)
- **Want details?** → `docs/frontend.md`
- **API help?** → `docs/api.md`
- **Full workflow?** → `docs/workflow.md`
- **Code changes?** → `docs/code_structure.md`
- **GitHub Issues** → https://github.com/ctftamu/SL-GPS/issues
- **Discord Help** → https://discord.com/channels/1333609076726431798/1333610748424880128

## File Structure Overview

```
SL-GPS/
├── frontend/              ← GUI code (python -m frontend)
│   ├── app.py            ← Main application
│   ├── requirements.txt   ← GUI dependencies
│   └── README.md          ← Detailed frontend docs
├── docs/
│   ├── frontend.md        ← GUI guide
│   ├── setup.md           ← Installation
│   ├── api.md             ← API reference
│   └── ...
├── FRONTEND_QUICKSTART.md ← Quick reference
├── IMPLEMENTATION_SUMMARY.md ← What was built
└── src/slgps/
    ├── main.py           ← Data generation (called by GUI)
    ├── mech_train.py     ← NN training (called by GUI)
    └── SL_GPS.py         ← Run simulations with trained model
```

## Success Checklist

- ✅ Installed Python 3.8+
- ✅ Cloned SL-GPS repository
- ✅ Installed dependencies (`pip install -r requirements.txt`)
- ✅ Installed frontend (`pip install -r frontend/requirements.txt`)
- ✅ Launched GUI (`python -m frontend`)
- ✅ Browser opened to http://localhost:7860
- ✅ Uploaded a mechanism file
- ✅ Generated a dataset
- ✅ Trained a neural network
- ✅ Found results in `generated_data/`

**All done!** You're now ready to use SL-GPS for chemistry reduction. 🎉

---

**Need Help?**
- 📖 Read the docs: `docs/frontend.md`
- 🔍 Quick reference: `FRONTEND_QUICKSTART.md`
- 💬 Discord: https://discord.com/channels/1333609076726431798/1333610748424880128
- 📧 Email: rmishra@tamu.edu

**Happy chemistry reduction!** 🧪
