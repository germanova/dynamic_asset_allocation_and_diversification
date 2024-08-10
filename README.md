# Dynamic Asset Allocation and Diversification Techniques

This project explores various methods for managing and optimizing asset portfolios using Python. The code and examples provided implement dynamic asset allocation and portfolio diversification strategies.

## Structure

### Python Files

- **`data_and_descriptives.py`**  
  This script is responsible for:
  - Generating descriptive statistics for the selected assets.
  - Downloading and preprocessing asset data.
  - Visualizing asset performance and characteristics.

- **`dynamic_asset_allocation.py`**  
  Contains implementations for dynamic asset allocation strategies, including:
  - **CPPI (Constant Proportion Portfolio Insurance)**
  - **MDD (Maximum Drawdown)**
  - **RDD (Relative Drawdown)**
  - **EDD (Expected Drawdown)**

- **`diversification.py`**  
  Focuses on portfolio diversification through:
  - Various rebalancing strategies.
  - Variants of Equal-Weighted (EW) rebalancing.
  - Incorporating stop-loss and take-profit mechanisms to manage risk and returns.

### Jupyter Notebooks

- **`compare_data.ipynb`**  
  Shows a basic example of the **`data_and_descriptives.py`** file functions in order to compare different assets.
- **`daa_diversification.ipynb`** 
  Shows one of the backtesting strategies based on DAA and diversification strategies.
- **`create_report.ipynb`** 
  Creates a report that outputs returns perfomance and risk metrics, along characteristics of the selected ETFs based on the **`etf_ib_data.xlsx`**  data.
### Data Files

- **`etf_ib_data.xlsx`**  
  Contains descriptive data about ETFs used in the analysis.

