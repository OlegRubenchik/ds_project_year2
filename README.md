# Data Science Project

## Description
This is a data science project created as part of the second year curriculum.

## Setup and usage
1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run, it will also generate infographics and save to data/insights: 
   ```bash
   python setup.py
   ```
5. Train the model:
   ```bash
   python models/price_reg.py
   ```
6. Use the model:
   ```bash
   python predict.py
   ```
