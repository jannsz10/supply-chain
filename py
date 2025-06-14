"""
Supply Chain Network Analysis Module.

This module generates a simulated supply chain network and performs network analysis
using NetworkX. The network includes suppliers, manufacturers, distributors, and retailers.
"""
#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class NodeAttributes:
    """Attributes for network nodes."""
    node_type: str
    capacity: float
    reliability: float
    location: Tuple[float, float]

class SupplyChainNetwork:
    """Supply Chain Network Generator and Analyzer."""

    def __init__(self, n_nodes: int = 500):
        """
        Initialize the supply chain network.

        Args:
            n_nodes: Total number of nodes in the network
        """
        self.n_nodes = n_nodes
        self.G = nx.DiGraph()
        self.node_types = ['supplier', 'manufacturer', 'distributor', 'retailer']
        self.type_distribution = {
            'supplier': 0.3,
            'manufacturer': 0.2,
            'distributor': 0.25,
            'retailer': 0.25
        }

    def generate_network(self) -> None:
        """Generate the supply chain network with nodes and edges."""
        # Generate nodes
        n_nodes_per_type = {
            node_type: int(self.n_nodes * prob)
            for node_type, prob in self.type_distribution.items()
        }

        # Adjust for rounding errors
        total = sum(n_nodes_per_type.values())
        if total < self.n_nodes:
            n_nodes_per_type['retailer'] += (self.n_nodes - total)

        # Create nodes with attributes
        node_id = 0
        for node_type, count in n_nodes_per_type.items():
            for _ in range(count):
                attributes = NodeAttributes(
                    node_type=node_type,
                    capacity=np.random.uniform(100, 1000),
                    reliability=np.random.uniform(0.7, 0.99),
                    location=(np.random.uniform(0, 100), np.random.uniform(0, 100))
                )
                self.G.add_node(node_id, **vars(attributes))
                node_id += 1

        # Generate edges based on network structure
        self._generate_edges()

    def _generate_edges(self) -> None:
        """Generate edges between nodes based on supply chain hierarchy."""
        nodes_by_type = defaultdict(list)
        for node, attrs in self.G.nodes(data=True):
            nodes_by_type[attrs['node_type']].append(node)

        # Define allowed connections
        allowed_connections = {
            'supplier': ['manufacturer'],
            'manufacturer': ['distributor'],
            'distributor': ['retailer'],
            'retailer': []
        }

        # Generate edges
        for source_type, target_types in allowed_connections.items():
            for target_type in target_types:
                source_nodes = nodes_by_type[source_type]
                target_nodes = nodes_by_type[target_type]

                # Calculate number of edges based on network density
                n_edges = int(len(source_nodes) * len(target_nodes) * 0.3)

                # Generate random edges
                for _ in range(n_edges):
                    source = np.random.choice(source_nodes)
                    target = np.random.choice(target_nodes)
                    weight = np.random.uniform(0.1, 1.0)
                    self.G.add_edge(source, target, weight=weight)

    def analyze_network(self) -> Dict:
        """
        Analyze the network and return key metrics.

        Returns:
            Dict containing network metrics
        """
        metrics = {
            'number_of_nodes': self.G.number_of_nodes(),
            'number_of_edges': self.G.number_of_edges(),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'density': nx.density(self.G),
            'average_clustering': nx.average_clustering(self.G.to_undirected()),
            'average_shortest_path_length': nx.average_shortest_path_length(self.G.to_undirected()),
            'node_type_distribution': defaultdict(int)
        }

        # Calculate node type distribution
        for node, attrs in self.G.nodes(data=True):
            metrics['node_type_distribution'][attrs['node_type']] += 1

        return metrics

    def visualize_network(self, save_path: str = 'supply_chain_network.png') -> None:
        """
        Visualize the network using matplotlib.

        Args:
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 15))

        # Set style
        sns.set_style("whitegrid")

        # Define node colors and positions
        node_colors = {
            'supplier': 'lightblue',
            'manufacturer': 'lightgreen',
            'distributor': 'orange',
            'retailer': 'pink'
        }

        # Get node positions using spring layout
        pos = nx.spring_layout(self.G, k=1, iterations=50)

        # Draw nodes
        for node_type in self.node_types:
            nodes = [n for n, attrs in self.G.nodes(data=True)
                    if attrs['node_type'] == node_type]
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=nodes,
                node_color=node_colors[node_type],
                node_size=100,
                alpha=0.7,
                label=node_type.capitalize()
            )

        # Draw edges
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=0.2,
            arrows=True,
            arrowsize=10
        )

        plt.title('Supply Chain Network Visualization', fontsize=16)
        plt.legend(fontsize=12)
        plt.axis('off')

        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate the supply chain network analysis."""
    # Create and generate network
    scn = SupplyChainNetwork(n_nodes=500)
    scn.generate_network()

    # Analyze network
    metrics = scn.analyze_network()

    # Print metrics
    print("\nSupply Chain Network Analysis Results:")
    print("-" * 40)
    for key, value in metrics.items():
        if key != 'node_type_distribution':
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")

    print("\nNode Type Distribution:")
    for node_type, count in metrics['node_type_distribution'].items():
        print(f"{node_type.capitalize()}: {count}")

    # Visualize network
    scn.visualize_network()

if __name__ == "__main__":
    main()

# %%
import pandas as pd
import folium

# --- 1. 加载数据 ---
try:
    nodes_df = pd.read_csv('nodes.csv')
    edges_df = pd.read_csv('edges.csv')
except FileNotFoundError:
    print("错误：请确保 'nodes.csv' 和 'edges.csv' 文件存在于当前目录。")
    # 为了演示，创建一些示例数据
    nodes_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['供应商A', '工厂X', '仓库Y', '客户Z', '供应商B'],
        'latitude': [34.0522, 36.7783, 40.7128, 39.9526, 35.6895],
        'longitude': [-118.2437, -119.4179, -74.0060, -75.1652, 139.6917],
        'type': ['供应商', '工厂', '仓库', '客户', '供应商'],
        'capacity': [1000, 5000, 10000, 0, 800]
    }
    edges_data = {
        'source_id': [1, 2, 3, 5],
        'target_id': [2, 3, 4, 2],
        'volume': [500, 2000, 1500, 300]
    }
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)
    print("示例数据已创建。")
#%%

# --- (可选) 地理编码 ---
# 如果你只有地址，没有经纬度，需要使用 geopy 进行转换。
# from geopy.geocoders import Nominatim
# geolocator = Nominatim(user_agent="supply_chain_mapper")
# def geocode_address(address):
#     try:
#         location = geolocator.geocode(address, timeout=10)
#         if location:
#             return location.latitude, location.longitude
#     except Exception as e:
#         print(f"地理编码错误: {e}")
#     return None, None
#
# # 假设 nodes_df 有 'address' 列
# # nodes_df[['latitude', 'longitude']] = nodes_df['address'].apply(
# #     lambda x: pd.Series(geocode_address(x))
# # )
# # nodes_df.dropna(subset=['latitude', 'longitude'], inplace=True) # 删除无法编码的行

# --- 2. 创建地图 ---
# 计算地图的中心点，可以取所有节点经纬度的平均值
map_center = [nodes_df['latitude'].mean(), nodes_df['longitude'].mean()]
supply_chain_map = folium.Map(location=map_center, zoom_start=4) # zoom_start 可调整默认缩放级别

# --- 3. 定义节点颜色和图标 (可选) ---
def get_node_color(node_type):
    if node_type == '供应商':
        return 'blue'
    elif node_type == '工厂':
        return 'red'
    elif node_type == '仓库':
        return 'orange'
    elif node_type == '客户':
        return 'green'
    else:
        return 'gray'

# --- 4. 在地图上绘制节点 ---
for idx, row in nodes_df.iterrows():
    popup_html = f"""
    <b>名称:</b> {row['name']}<br>
    <b>类型:</b> {row['type']}<br>
    <b>ID:</b> {row['id']}<br>
    <b>产能:</b> {row.get('capacity', 'N/A')}
    """
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_html, max_width=300), # 弹出窗口显示详细信息
        tooltip=row['name'], # 鼠标悬停提示
        icon=folium.Icon(color=get_node_color(row['type']), icon='industry' if row['type'] == '工厂' else 'info-sign')
    ).add_to(supply_chain_map)

    # 也可以使用 CircleMarker
    # folium.CircleMarker(
    #     location=[row['latitude'], row['longitude']],
    #     radius=5, # 可以根据节点的某个属性（如容量）动态调整大小
    #     color=get_node_color(row['type']),
    #     fill=True,
    #     fill_color=get_node_color(row['type']),
    #     fill_opacity=0.7,
    #     popup=f"{row['name']} ({row['type']})"
    # ).add_to(supply_chain_map)


# --- 5. 在地图上绘制连接 (边) ---
# 为了方便查找节点坐标，创建一个id到坐标的映射
node_coords = {row['id']: (row['latitude'], row['longitude']) for idx, row in nodes_df.iterrows()}

for idx, row in edges_df.iterrows():
    source_node = node_coords.get(row['source_id'])
    target_node = node_coords.get(row['target_id'])

    if source_node and target_node:
        # 创建连接线的坐标列表
        points = [source_node, target_node]
        # 线的提示信息
        edge_tooltip = f"从 {nodes_df[nodes_df['id']==row['source_id']]['name'].iloc[0]} 到 {nodes_df[nodes_df['id']==row['target_id']]['name'].iloc[0]}<br>运输量: {row.get('volume', 'N/A')}"

        folium.PolyLine(
            locations=points,
            color="purple", # 连接线颜色
            weight=max(1, row.get('volume', 100) / 500), # 可以根据运输量调整线的粗细
            opacity=0.7,
            tooltip=edge_tooltip
        ).add_to(supply_chain_map)

# --- 6. 保存地图 ---
supply_chain_map.save("supply_chain_map.html")
print("地图已保存为 'supply_chain_map.html'")

# 如果你在 Jupyter Notebook 中运行，可以直接显示地图：
# supply_chain_map
# %%
