import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
#Constants
# Set your city and radius here
CITY_NAME = 'Pretoria, South Africa'
RADIUS_DEGREE = 0.01  # ~2km radius

# Load graph
polygon = ox.geocode_to_gdf(CITY_NAME, which_result=1).geometry.iloc[0].buffer(RADIUS_DEGREE)
G = ox.graph_from_polygon(polygon, network_type="drive")

# ---------------------------- #
# Helper function to get meaningful edge name
# ---------------------------- #
GENERIC_NAMES = {'residential', 'service', 'footway', 'unclassified', 'track', 'path','yes', 'no', ''}

def get_edge_name(edge_data):
    # Try 'name' first
    name = edge_data.get('name')
    if isinstance(name, (list, tuple, np.ndarray)):
        name = ", ".join([str(n) for n in name if n])
    if name and not pd.isna(name) and name.lower() not in GENERIC_NAMES:
        return name
    
    # Try 'ref'
    ref = edge_data.get('ref')
    if isinstance(ref, (list, tuple, np.ndarray)):
        ref = ", ".join([str(r) for r in ref if r])
    if ref and not pd.isna(ref) and ref.lower() not in GENERIC_NAMES:
        return ref
    
    highway = edge_data.get('highway')
    if isinstance(highway, list):
        highway = highway[0]
    if isinstance(highway, str):
        if highway.lower().endswith("_link"):
            # follow the link to the next node and get its meaningful edge
            for _, next_v, k, next_edge in G.edges(v, keys=True, data=True):
                next_name = get_edge_name(next_edge)
                if next_name and next_name.lower() not in GENERIC_NAMES:
                    return next_name
            # fallback if next street not found
            highway = highway.replace("_link", "")
        if highway.lower() not in GENERIC_NAMES:
            return highway
        
    # Try other tags
    for key in ['bridge', 'footway', 'railway', 'secondary', 'tertiary', 'primary']:
        value = edge_data.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            value = ", ".join([str(v) for v in value if v])
        if isinstance(value, str) and value.strip() and value.lower() not in GENERIC_NAMES:
            return value
    
    return None


# Extract intersections with edge info
# Raw data from OSM, this would have more than 2 edges per intersection
data_rows = []
node_id_counter = 1

for nid, node_data in G.nodes(data=True):
    incident_edges = list(G.edges(nid, keys=True, data=True)) + list(G.in_edges(nid, keys=True, data=True))
    
    edge_list = []
    for u, v, k, edge_data in incident_edges:
        street_name = get_edge_name(edge_data)
        if not street_name:
            continue
        if u == v:
            direction = 'B'
        elif u == nid:
            direction = 'O'
        elif v == nid:
            direction = 'I'
        else:
            direction = 'B'
        edge_list.append(f"{street_name} ({direction})")
    
    if len(edge_list) >= 2:
        data_rows.append({
            "UniqueID": node_id_counter,
            "OSM_NodeID": nid,
            "NumEdges": len(edge_list),
            "Edges": "; ".join(edge_list),
            "X": node_data.get('x'),
            "Y": node_data.get('y')
        })
        node_id_counter += 1

intersections_df = pd.DataFrame(data_rows)
intersections_df.to_csv("intersections_formatted.csv", index=False)
print("Intersections CSV saved: intersections_formatted.csv")


# Movement rules
# I = Incoming, O = Outgoing, B = Both
# This function parses the edge string into a list of (name, direction) tuples
def parse_edges(edge_str):
    edges = []
    for e in edge_str.split(";"):
        e = e.strip()
        if not e:
            continue
        if "(" in e and ")" in e:
            name = e[:e.rfind("(")].strip()
            direction = e[e.rfind("(")+1:e.rfind(")")]
            edges.append((name, direction))
        else:
            edges.append((e.strip(), "B"))
    return edges

def valid_move(src_dir, dst_dir):
    if src_dir == "O" and dst_dir == "I": return False
    if src_dir == "I" and dst_dir == "I": return False
    if src_dir == "I" and dst_dir == "O": return True
    if src_dir == "B" and dst_dir == "B": return True
    if src_dir == "B" and dst_dir == "O": return True
    if src_dir == "I" and dst_dir == "B": return True
    return False

# Generate intersections 
# Each valid (src, dst) pair at an intersection becomes a unique node

node_mapping = {}  # (osm_node_id, src, src_dir, dst, dst_dir) -> cpp_node_id
cpp_id_counter = 1
output_nodes = []

for _, row in intersections_df.iterrows():
    osm_id = row.OSM_NodeID
    edges = parse_edges(row.Edges)
    x, y = row.Y, row.X

    for i in range(len(edges)):
        for j in range(len(edges)):
            if i == j: continue
            src, src_dir = edges[i]
            dst, dst_dir = edges[j]
            if not valid_move(src_dir, dst_dir): continue
            if src == dst: continue  # skip self-loop at same intersection

            key = (osm_id, src, src_dir, dst, dst_dir)
            if key not in node_mapping:
                node_mapping[key] = cpp_id_counter
                cpp_id_counter += 1
                # Strip direction from street name for C++ output
                src_name = src
                dst_name = dst
                output_nodes.append(f'intersections.insertNode("{src_name}", "{dst_name}", {x:.7f}, {y:.7f});')

# Generate C++ edges

osm_to_cpp_nodes = {}
for key, cpp_id in node_mapping.items():
    osm_node_id = key[0]
    osm_to_cpp_nodes.setdefault(osm_node_id, []).append(cpp_id)



# Create CSV for C++ nodes

cpp_rows = []
for key, cpp_id in node_mapping.items():
    osm_id, src, src_dir, dst, dst_dir = key
    row = intersections_df[intersections_df['OSM_NodeID'] == osm_id].iloc[0]
    cpp_rows.append({
        "CPP_ID": cpp_id,
        "OSM_NodeID": osm_id,
        "SrcStreet": src,
        "SrcDir": src_dir,
        "DstStreet": dst,
        "DstDir": dst_dir,
        "X": row['X'],
        "Y": row['Y']
    })
cpp_intersections_df = pd.DataFrame(cpp_rows)
cpp_intersections_df.to_csv("intersections_cpp.csv", index=False)
print("C++ intersections CSV saved: intersections_cpp.csv")


# Create CSV for C++ edges

edges_cpp_rows = []
for u, v, k, edge_data in G.edges(keys=True, data=True):
    if u in osm_to_cpp_nodes and v in osm_to_cpp_nodes:
        dist_km = edge_data.get('length', 0)
        for src_cpp in osm_to_cpp_nodes[u]:
            for dst_cpp in osm_to_cpp_nodes[v]:
                edges_cpp_rows.append({
                    "SRC_CPP_ID": src_cpp,
                    "DST_CPP_ID": dst_cpp,
                    "Distance_km": dist_km
                })
edges_cpp_df = pd.DataFrame(edges_cpp_rows)
edges_cpp_df.to_csv("edges_cpp.csv", index=False)
print("C++ edges CSV saved: edges_cpp.csv")

# --------------------------------------------------------
# MERGE STEP: Assign MergeIDs for compressed C++ output
# --------------------------------------------------------
merge_map = {osm_id: mid for mid, osm_id in enumerate(cpp_intersections_df["OSM_NodeID"].unique(), start=1)}
cpp_intersections_df["MergeID"] = cpp_intersections_df["OSM_NodeID"].map(merge_map)
# Build CPP_ID → MergeID mapping
cpp_to_merge = dict(zip(cpp_intersections_df['CPP_ID'], cpp_intersections_df['MergeID']))

# Generate edges using MergeID and remove duplicates
edge_set = set()
edge_rows = []

for u, v, k, edge_data in G.edges(keys=True, data=True):
    if u in osm_to_cpp_nodes and v in osm_to_cpp_nodes:
        dist_km = edge_data.get('length', 0)
        for src_cpp in osm_to_cpp_nodes[u]:
            for dst_cpp in osm_to_cpp_nodes[v]:
                src_merge = cpp_to_merge[src_cpp]
                dst_merge = cpp_to_merge[dst_cpp]
                if src_merge == dst_merge:
                    continue  # skip self-loop after merge
                if (src_merge, dst_merge) not in edge_set:
                    edge_set.add((src_merge, dst_merge))
                    edge_rows.append(f"    g.getEdgesMutable({src_merge}).push_back({{{dst_merge}, {dist_km:.7f}}});")



# Plot using C++ node IDs

intersections_df = pd.read_csv("intersections_cpp.csv")
edges_df = pd.read_csv("edges_cpp.csv")
id_to_coords = dict(zip(intersections_df['CPP_ID'], zip(intersections_df['X'], intersections_df['Y'])))

plt.figure(figsize=(12, 10))
plt.scatter(intersections_df['X'], intersections_df['Y'], color='red', s=25, zorder=5)

for _, row in intersections_df.iterrows():
    plt.text(row['X'], row['Y'], str(row['CPP_ID']), fontsize=6, color='blue', zorder=10)

for _, edge in edges_df.iterrows():
    src_id = edge['SRC_CPP_ID']
    dst_id = edge['DST_CPP_ID']
    if src_id in id_to_coords and dst_id in id_to_coords:
        x_values = [id_to_coords[src_id][0], id_to_coords[dst_id][0]]
        y_values = [id_to_coords[src_id][1], id_to_coords[dst_id][1]]
        plt.plot(x_values, y_values, color='gray', linewidth=0.7, zorder=1)

plt.title(CITY_NAME + " Intersections with Edges", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("intersections_with_edges_cpp.png", dpi=300)
plt.show()
print("PNG saved: intersections_with_edges_cpp.png")
