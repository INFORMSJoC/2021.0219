# List of Data Folders

"Generator_Data" is a folder with the data of generators used for Tables 2 - 3 of the paper.

"Data_for_Tables_2-3" is a folder with randomly generated instances for Tables 2 - 3 of the paper.

"Data_for_Table_4" is a folder with nominal data and python code to generate random instances for Table 4 of the paper.

# Details of Each Folder

"Generator_Data" folder:

The files G1.dat, G2.dat, and G3.dat contain the physical data of three natural gas-fired generators, which are mentioned in Table 1 of the paper.

"Data_for_Tables_2-3" folder:

Each file contains a randomly generated instance used to generate the results in Tables 2 - 3. The filename of each file follows the same following format: (i) G stands for the order of the generator; (ii) R stands for the number of stages in the planning horizon, where each stage represents one day; (iii) SceNo stands for the number of scenarios in the scenario tree; (iv) Inst stands for an instance. For example, G1_R6_SceNo16_Inst1.csv contains the data for the first instance considering the generator G1 and six stages and 16 scenarios in the scenario tree. In each file, the electricity price and the fuel supply corresponding to each scenario node (represented by the scenario node id) in the scenario tree are provided.

"Data_for_Table_4" folder:

The python file generate_data.py generates random instances and constructs the model data. The files electricity_price_data_21.csv, electricity_price_data_26.csv, and electricity_price_data_31.csv contain the nominal electricity price used to generate random electricity price data. All these generated data will be input into the SDDiP program available at a GitHub Repository (see https://github.com/brownflaming/SDDiP), provided by the first author of the paper "Zou, Jikai, Shabbir Ahmed, and Xu Andy Sun. 'Stochastic dual dynamic integer programming.' Mathematical Programming 175.1 (2019): 461-502."
