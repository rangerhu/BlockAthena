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

## 📊 Real-world Case Result

### Case 1️⃣: AscenDEX Hack
On December 11, 2021, the centralized cryptocurrency exchange AscenDEX experienced a major security breach.
BlockAthena first visualizes the long-term  crime-related behavioral traits MMTS as follows: 
![AscenDEX Crime Evolution MMTS](realworld_case_results/ascendex/MMTS.png)
Then, the figure below illustrates the crime evolution periods identified in the AscenDEX hack using BlockAthena:
![AscenDEX Crime Evolution](realworld_case_results/ascendex/main_period_trends.png)


### Case 2️⃣: KuCoin Hack
On September 25, 2020, the centralized cryptocurrency exchange KuCoin experienced a major security breach involving the compromise of its hot wallets.
BlockAthena first visualizes the long-term  crime-related behavioral traits MMTS  as follows: 
![KuCoin Hack Evolution MMTS](realworld_case_results/kucoin/MMTS.png)
Then, the figure below illustrates the crime evolution periods identified in the KuCoin Hack using BlockAthena:
![KuCoin Hack Evolution](realworld_case_results/kucoin/main_period_trends.png)


### Case 3️⃣: WarpFinance Hack
On December 17, 2020, WarpFinance—a DeFi protocol for stablecoin lending—was exploited in a sophisticated flash loan attack.
BlockAthena first visualizes the long-term  crime-related behavioral traits MMTS as  follows: 
![WarpFinance Hack Evolution MMTS](realworld_case_results/warpfinance/MMTS.png)
Then, the figure below illustrates the crime evolution periods identified in the WarpFinance Hack using BlockAthena:
![WarpFinance Hack Evolution](realworld_case_results/warpfinance/main_period_trends.png)


## 🚀 Getting Started

### Step 1️⃣: Prepare Ethereum Transaction Data

Collect transaction data of Ethereum crime-related addresses. You can obtain this data from:

- e.g., Ethereum full node  


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


