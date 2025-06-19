# BlockAthena: A Scalable Approach for Long-Term Ethereum Crimes Analysis

This project implements BlockAthena, a scalable framework fo long-term regulatory analysis of ever-growing transaction topology in Ethereum blockchain. BlockAthena ensures both memory efficiency and effective mining of evolving crime patterns, even stealthy crime tactics such as APT-like life-cycles and botnet-style behaviors.

## 📌 Key Components

1. **Scalable modeling of evolution periods**  
   A scalable temporal evolution modeling method named Motif-aware Periodicity Modeling (MPM) to formulate transaction subgraphs at each evolution period of Ethereum crimes by jointly modeling and analyzing  transactional patterns of Ethereum crimes in the time-frequency-topology domain.

2. **Mixed-order behavioral feature mining**  
   A mixed-order behavioral analyzer that profiles both low-order and high-order transactional interactions within each evolution period, enabling comprehensive mining of crime-related behavior features.

3. **Adaptive long-term pattern synthesis**  
   A Evolution-aware Residual Aggregator (ERA), which synthesizes long-term trends of Ethereum crimes across segmented evolution periods.

4. **Visualization**  
   Generates plots including motif trends and wavelet heatmaps with marked evolution periods.

## 🛠️ Directory Structure

```
BlockAthena/
├── config.py                    # Global configuration parameters
├── main_runner.py              # Pipeline entry point
├── mpm/                        # Motif-aware Periodicity Modeling
│   ├── motif_modeler.py
│   ├── wavelet_analyzer.py
│   └── subgraph_segmenter.py
├── behavior/                   # Mixed-order Behavioral Feature Mining
│   └── mixed_order_analyzer.py
├── era/                        # ERA: Adaptive Long-term Pattern Synthesis
│   └── era_aggregator.py
├── utils/                      # Utility functions
│   ├── io_utils.py
│   ├── time_utils.py
│   └── graph_utils.py
├── visualization/             # Visualization helpers
│   ├── motif_plot.py
│   └── wavelet_plot.py
├── data/
│   ├── raw/                    # Input transactions 
│   ├── processed/              
├── realworld_case_results/     # Results of real-world case study 
│   ├── ascendex/               # AscenDEX hack 
│   ├── kucoin/                 # KuCoin hack 
│   ├── warpfinance/            # WarpFinance hack
└── README.md                  
```

## 🚀 Getting Started

### Step 1️⃣: Prepare Ethereum Transaction Data

Collect transaction data of Ethereum crime-related addresses. You can obtain this data from:

- Ethereum full node, or  
- Public blockchain explorers (e.g., Etherscan APIs)

The transaction CSV should include at least the following columns:

- `from`: sender address  
- `to`: receiver address  
- `timestamp`: Unix timestamp

Place the file in the following directory: data/raw/


### Step 2️⃣: Install Dependencies

Before running the pipeline, make sure all required Python dependencies are installed.

You can install them using `pip`:

```bash
pip install -r requirements.txt
```



### Step 3️⃣: Run the BlockAthena Pipeline

Execute the full pipeline by running:

```bash
python main_runner.py --input data/raw/transactions.csv --target example_case --outdir data/processed
```


















## 📄 Dependencies
```
pandas
numpy
matplotlib
networkx
pywt
scipy
torch
torch_geometric
```

