import pandas as pd
from teneto import TemporalNetwork
from teneto.networkmeasures import temporal_betweenness_centrality
import sys

def load_df(filename):
    df = pd.read_csv(filename, header=None, names=['u', 'v', 'w', 't'])
    df = df[['u', 'v', 't']]  # drop weight
    df.columns = ['i', 'j', 't']  # Teneto expects columns named i, j, t
    return df

def main(csv_file):
    df = load_df(csv_file)
    # Build the temporal network from DataFrame
    tnet = TemporalNetwork(from_df=df, nettype='bd', timetype='discrete')  # bi-directed (bd)
    # Compute normalized betweenness centrality (averaged over time)
    bc = temporal_betweenness_centrality(tnet, calc='overtime')
    print("Normalized temporal BC:")
    for i, score in enumerate(bc):
        print(f"BC[{i}] = {score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_bc_csv.py <your_edges.csv>")
        sys.exit(1)
    main(sys.argv[1])
