import json
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ---------------- Data Loading ----------------

def load_dataset(json_path, dist_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Distance dictionary
    dist = {}
    with open(dist_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            origin, dest, _, meters = parts
            key = (origin, dest)
            dist[key] = float(meters) / 1000.0  # convert to km

    return data, dist


def build_distance_matrix(nodes, dist_dict):
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    matrix = [[0.0] * n for _ in range(n)]
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            matrix[i][j] = dist_dict.get((ni, nj), dist_dict.get((nj, ni), 0.0))
    return matrix, index


def group_orders_by_destination(orders):
    dest_map = {}
    for o in orders:
        dest = o['destination']
        dest_map.setdefault(dest, []).append(o)
    return dest_map


# ---------------- VRP Solver ----------------


def solve_vrp(distance_matrix, demands, vehicle_cap):
    n = len(distance_matrix)
    depot = 0
    manager = pywrapcp.RoutingIndexManager(n, 999, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(distance_matrix[i][j] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_cb(index):
        node = manager.IndexToNode(index)
        return demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [vehicle_cap] * routing.vehicles(),
        True,
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)
    routes = []
    if solution:
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)
    return routes


# ---------------- Packing ----------------


def pack_boxes(routes, dest_orders, node_list, vehicle_dim):
    result = []
    veh_id = 0
    width, length, height = vehicle_dim
    for route in routes:
        y_pos = 0
        stacking_order = 1
        # reverse exclude depot at start and end
        seq = [node_list[i] for i in route[1:-1]][::-1]
        for dest in seq:
            orders = dest_orders[dest]
            for o in orders:
                result.append({
                    'Vehicle_ID': veh_id,
                    'Route_Order': None,
                    'Destination': dest,
                    'Order_Number': o['order_number'],
                    'Box_ID': o['box_id'],
                    'Stacking_Order': stacking_order,
                    'Lower_Left_X': 0,
                    'Lower_Left_Y': y_pos,
                    'Lower_Left_Z': 0,
                    'Box_Width': o['dimension']['width'],
                    'Box_Length': o['dimension']['length'],
                    'Box_Height': o['dimension']['height'],
                })
                stacking_order += 1
                y_pos += o['dimension']['length']
        veh_id += 1
    return result


def main():
    data, dist = load_dataset('problem-docs/Data_Set.json', 'problem-docs/distance-data.txt')
    depot = 'Depot'
    destinations = [d['destination_id'] for d in data['destinations']]
    nodes = [depot] + destinations

    matrix, idx = build_distance_matrix(nodes, dist)

    orders = data['orders']
    dest_orders = group_orders_by_destination(orders)
    demands = [0]
    for dest in destinations:
        total_vol = sum(o['dimension']['width'] * o['dimension']['length'] * o['dimension']['height'] for o in dest_orders.get(dest, []))
        demands.append(total_vol)

    vehicle_cap = data['vehicles'][0]['dimension']['width'] * data['vehicles'][0]['dimension']['length'] * data['vehicles'][0]['dimension']['height']
    routes = solve_vrp(matrix, demands, vehicle_cap)

    packing = pack_boxes(routes, dest_orders, nodes, (
        data['vehicles'][0]['dimension']['width'],
        data['vehicles'][0]['dimension']['length'],
        data['vehicles'][0]['dimension']['height']))

    # assign route order per vehicle
    route_orders = {}
    for veh_idx, route in enumerate(routes):
        order = 1
        for node in route:
            dest = nodes[node]
            route_orders.setdefault(veh_idx, []).append({'order': order, 'dest': dest})
            order += 1

    # merge route order with packing
    for p in packing:
        veh = p['Vehicle_ID']
        # find index of dest in route
        dest = p['Destination']
        rlist = route_orders[veh]
        for item in rlist:
            if item['dest'] == dest:
                p['Route_Order'] = item['order']
                break

    df = pd.DataFrame(packing)
    df = df[['Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
             'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
             'Box_Width', 'Box_Length', 'Box_Height']]
    df.to_excel('Result.xlsx', index=False, sheet_name='Detailed Route Information')


if __name__ == '__main__':
    main()
