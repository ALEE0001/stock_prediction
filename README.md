# Stock Prediction Platform

## Overview
This repository is a comprehensive platform for stock price prediction and backtesting, designed for artificial intelligence and data science applications. It leverages advanced machine learning models, including deep learning architectures, to forecast stock movements and evaluate trading strategies. The project is structured to facilitate experimentation, model development, and performance analysis, making it ideal for showcasing AI and data science skills to recruiters.

## Key Features
- **End-to-End Pipeline:** From data ingestion and preprocessing to model training, evaluation, and backtesting.
- **Multi-Task Learning:** Implements custom models (e.g., Temporal Fusion Transformer) for simultaneous regression and classification tasks.
- **Backtesting Framework:** Evaluate trading strategies with detailed performance reports and visualizations.
- **Jupyter Notebooks:** Interactive notebooks for exploratory data analysis, model experimentation, and results visualization.
- **Modular Codebase:** Organized into scripts and modules for easy extension and maintenance.

## Repository Structure
```
├── backtesting.ipynb           # Backtesting and strategy evaluation notebook
├── manual.ipynb                # Manual experimentation and notes
├── model_build.py              # Model building and training script
├── model_classification.ipynb  # Classification model experiments
├── model_regression.ipynb      # Regression model experiments
├── playground.ipynb            # Main experimentation notebook
├── README.md                   # Project documentation (this file)
├── secrets.json                # API keys and sensitive info (excluded from version control)
├── yt.py                       # YouTube or data utility script
├── archive/                    # Archived scripts and experiments
│   └── model_train.py          # Legacy or experimental model training code
├── data/                       # Data directory
│   ├── data_model.csv          # Processed dataset for modeling
│   ├── data.csv                # Raw or intermediate data
│   ├── fred.csv                # Economic indicators (FRED)
│   └── raw/                    # Raw data files
├── logs/                       # Backtest results, logs, and reports
│   ├── *_settings.json         # Backtest configuration
│   ├── *_tearsheet.html        # Performance reports
│   ├── *_trades.csv            # Trade logs
│   └── *_trades.html           # Trade visualizations
├── models/                     # Saved models and checkpoints
├── src/                        # Source code for models and utilities
```

## Notable Technologies
- **Python** (Pandas, PyTorch, PyTorch Forecasting, scikit-learn)
- **Jupyter Notebooks** for interactive development
- **Machine Learning & Deep Learning** (custom loss functions, multi-task models)
- **Backtesting & Performance Analysis**

## How to Use
1. **Data Preparation:** Place your data in the `data/` directory. Use the provided scripts/notebooks for preprocessing.
2. **Model Training:** Use `model_build.py` or the relevant notebooks to train and evaluate models.
3. **Backtesting:** Run `backtesting.ipynb` to simulate trading strategies and analyze results.
4. **Experimentation:** Use `playground.ipynb` for custom experiments and prototyping.

## Recruiter Highlights
- Demonstrates expertise in AI/ML model development, including custom architectures and loss functions.
- Showcases ability to build robust, modular, and production-ready data science pipelines.
- Includes real-world backtesting and performance reporting for financial applications.
- Code is well-organized, documented, and ready for extension or integration into larger systems.

## Contact
For more information or to discuss this project, please contact the repository owner via LinkedIn or email (see GitHub profile).