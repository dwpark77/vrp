import json
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import permutations
import time
import sys

class VRPSolver:
    def __init__(self, data_file, distance_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.distance_matrix = self.load_distance_matrix(distance_file)
        
        # 차량 제원
        self.truck_capacity = (160, 280, 180)  # cm (w, d, h)
        self.fixed_cost = 150000  # 원
        self.fuel_cost_per_km = 500  # 원/km
        self.shuffle_cost = 500  # 원/회
        
        # 박스 크기 종류
        self.box_types = {
            1: (30, 40, 30),
            2: (30, 50, 40), 
            3: (50, 60, 50)
        }
        
        self.depot = self.data['depot']
        self.destinations = {d['destination_id']: d for d in self.data['destinations']}
        self.orders = self.data['orders']
        
    def load_distance_matrix(self, distance_file):
        """거리 행렬 로드"""
        distances = {}
        with open(distance_file, 'r') as f:
            next(f)  # 헤더 스킵
            for line in f:
                parts = line.strip().split('\t')
                origin, dest, time_min, distance_m = parts
                distances[(origin, dest)] = int(distance_m) / 1000  # km 변환
        return distances
    
    def get_distance(self, from_id, to_id):
        """두 지점 간 거리 반환"""
        if from_id == "Depot":
            from_id = "Depot"
        if to_id == "Depot":
            to_id = "Depot"
            
        if (from_id, to_id) in self.distance_matrix:
            return self.distance_matrix[(from_id, to_id)]
        elif (to_id, from_id) in self.distance_matrix:
            return self.distance_matrix[(to_id, from_id)]
        else:
            return 999999  # 매우 큰 값
    
    def calculate_route_cost(self, route):
        """경로 비용 계산"""
        if not route:
            return 0
        
        total_distance = 0
        current = "Depot"
        
        for dest in route:
            total_distance += self.get_distance(current, dest)
            current = dest
        
        # 마지막에 창고로 돌아가기
        total_distance += self.get_distance(current, "Depot")
        
        return self.fixed_cost + total_distance * self.fuel_cost_per_km
    
    def can_fit_in_truck(self, orders_subset):
        """주문들이 트럭에 들어갈 수 있는지 확인"""
        total_volume = 0
        truck_volume = self.truck_capacity[0] * self.truck_capacity[1] * self.truck_capacity[2]
        
        for order in orders_subset:
            box_id = order['box_id']
            if box_id in self.box_types:
                box_vol = np.prod(self.box_types[box_id])
                total_volume += box_vol
        
        return total_volume <= truck_volume
    
    def calculate_shuffle_cost(self, orders_in_truck):
        """셔플링 비용 계산 (간소화 버전)"""
        dest_groups = defaultdict(list)
        for order in orders_in_truck:
            dest_groups[order['destination']].append(order)
        
        # 간단한 셔플링 추정: 목적지 수에 비례
        num_destinations = len(dest_groups)
        if num_destinations <= 1:
            return 0
        
        # 평균적으로 목적지당 몇 개의 박스를 다시 옮겨야 하는지 추정
        avg_shuffle_per_dest = max(0, num_destinations - 1) * 0.5
        total_shuffle = avg_shuffle_per_dest * num_destinations
        
        return int(total_shuffle) * self.shuffle_cost
    
    def greedy_vrp_solve(self):
        """그리디 알고리즘으로 VRP 해결 (최적화 버전)"""
        # 목적지별로 주문 그룹화
        dest_orders = defaultdict(list)
        for order in self.orders:
            dest_orders[order['destination']].append(order)
        
        destinations = list(dest_orders.keys())
        vehicles = []
        visited_dests = set()
        
        while len(visited_dests) < len(destinations):
            current_vehicle_orders = []
            current_destinations = []
            current_volume = 0
            truck_volume = self.truck_capacity[0] * self.truck_capacity[1] * self.truck_capacity[2]
            
            # 가장 가까운 목적지부터 추가
            current_pos = "Depot"
            
            while True:
                best_dest = None
                best_distance = float('inf')
                
                for dest in destinations:
                    if dest in visited_dests or dest in current_destinations:
                        continue
                    
                    # 용량 체크
                    dest_volume = sum(np.prod(self.box_types[order['box_id']]) 
                                    for order in dest_orders[dest]
                                    if order['box_id'] in self.box_types)
                    
                    if current_volume + dest_volume > truck_volume:
                        continue
                    
                    distance = self.get_distance(current_pos, dest)
                    if distance < best_distance:
                        best_distance = distance
                        best_dest = dest
                
                if best_dest is None:
                    break
                
                # 목적지 추가
                current_destinations.append(best_dest)
                current_vehicle_orders.extend(dest_orders[best_dest])
                dest_volume = sum(np.prod(self.box_types[order['box_id']]) 
                                for order in dest_orders[best_dest]
                                if order['box_id'] in self.box_types)
                current_volume += dest_volume
                current_pos = best_dest
            
            if current_vehicle_orders:
                vehicles.append(current_vehicle_orders)
                visited_dests.update(current_destinations)
        
        return vehicles
    
    def optimize_loading_order(self, orders_in_truck):
        """트럭 내 적재 순서 최적화"""
        dest_groups = defaultdict(list)
        for order in orders_in_truck:
            dest_groups[order['destination']].append(order)
        
        destinations = list(dest_groups.keys())
        
        # 목적지가 적으면 모든 순열 확인
        if len(destinations) <= 6:
            best_cost = float('inf')
            best_route = destinations
            
            for route in permutations(destinations):
                route_cost = self.calculate_route_cost(list(route))
                shuffle_cost = self.calculate_shuffle_cost(orders_in_truck)
                total_cost = route_cost + shuffle_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route
            
            return list(best_route)
        else:
            # 너무 많으면 그리디 접근
            return destinations
    
    def solve(self):
        """VRP 문제 해결"""
        start_time = time.time()
        
        vehicles = self.greedy_vrp_solve()
        
        total_cost = 0
        solution = []
        
        for i, vehicle_orders in enumerate(vehicles):
            # 목적지 추출
            destinations = list(set(order['destination'] for order in vehicle_orders))
            
            # 경로 최적화
            optimized_route = self.optimize_loading_order(vehicle_orders)
            
            # 비용 계산
            route_cost = self.calculate_route_cost(optimized_route)
            shuffle_cost = self.calculate_shuffle_cost(vehicle_orders)
            vehicle_total = route_cost + shuffle_cost
            
            total_cost += vehicle_total
            
            solution.append({
                'vehicle_id': f'V_{i+1:03d}',
                'orders': vehicle_orders,
                'route': optimized_route,
                'route_cost': route_cost,
                'shuffle_cost': shuffle_cost,
                'total_cost': vehicle_total
            })
        
        end_time = time.time()
        
        return {
            'vehicles': solution,
            'total_cost': total_cost,
            'execution_time': end_time - start_time
        }
    
    def pack_boxes_in_truck(self, orders):
        """트럭 내 박스 3D 배치"""
        # 목적지별로 그룹화하고 배송 순서 결정
        dest_groups = defaultdict(list)
        for order in orders:
            dest_groups[order['destination']].append(order)
        
        # 배송 순서 (거리 기반)
        delivery_route = self.optimize_loading_order(orders)
        
        # LIFO를 위해 적재는 배송 순서의 역순
        loading_order = delivery_route[::-1]
        
        packed_boxes = []
        layer_height = 0  # 현재 층의 높이
        current_x, current_y = 0, 0
        current_z = 0
        max_height_in_layer = 0
        
        stacking_order = len([o for orders_list in dest_groups.values() for o in orders_list])
        
        for dest in loading_order:
            dest_orders = dest_groups[dest]
            
            for order in dest_orders:
                box_id = order['box_id']
                if box_id in self.box_types:
                    w, l, h = self.box_types[box_id]
                else:
                    w, l, h = order['dimension']['width'], order['dimension']['length'], order['dimension']['height']
                
                # 현재 위치에 박스가 들어갈 수 있는지 확인
                if current_x + w > self.truck_capacity[0]:
                    # 다음 줄로 이동
                    current_x = 0
                    current_y += 50  # 일정 간격으로 Y 증가
                    
                    # Y 좌표가 트럭 깊이를 초과하면 새 층으로
                    if current_y + l > self.truck_capacity[1]:
                        current_y = 0
                        current_z += max_height_in_layer if max_height_in_layer > 0 else 60
                        max_height_in_layer = 0
                
                # 박스 배치
                packed_boxes.append({
                    'order': order,
                    'destination': dest,
                    'x': current_x,
                    'y': current_y,
                    'z': current_z,
                    'width': w,
                    'length': l,
                    'height': h,
                    'stacking_order': stacking_order
                })
                
                # 좌표 업데이트
                current_x += w
                max_height_in_layer = max(max_height_in_layer, h)
                stacking_order -= 1
        
        return packed_boxes
    
    def save_result(self, solution, output_file='Result.xlsx'):
        """결과를 Excel 파일로 저장 (요구된 형식)"""
        results = []
        
        for i, vehicle in enumerate(solution['vehicles']):
            vehicle_id = i
            
            # Depot 시작점 추가
            results.append({
                'Vehicle_ID': vehicle_id,
                'Route_Order': 1,
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': self.depot['location']['longitude'],
                'Latitude': self.depot['location']['latitude'],
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            })
            
            # 박스 배치 계산
            packed_boxes = self.pack_boxes_in_truck(vehicle['orders'])
            
            # 목적지별로 정렬
            dest_order = {}
            route_order = 2
            for dest in vehicle['route']:
                if dest not in dest_order:
                    dest_order[dest] = route_order
                    route_order += 1
            
            for packed_box in packed_boxes:
                order = packed_box['order']
                dest = packed_box['destination']
                dest_info = self.destinations[dest]
                
                results.append({
                    'Vehicle_ID': vehicle_id,
                    'Route_Order': dest_order[dest],
                    'Destination': dest,
                    'Order_Number': order['order_number'],
                    'Box_ID': order['box_id'],
                    'Stacking_Order': packed_box['stacking_order'],
                    'Lower_Left_X': packed_box['x'],
                    'Lower_Left_Y': packed_box['y'],
                    'Lower_Left_Z': packed_box['z'],
                    'Longitude': dest_info['location']['longitude'],
                    'Latitude': dest_info['location']['latitude'],
                    'Box_Width': packed_box['width'],
                    'Box_Length': packed_box['length'],
                    'Box_Height': packed_box['height']
                })
        
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)
        print(f"결과 파일 저장: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("사용법: python vrp_solver.py <data_file.json> <distance_file.txt>")
        return
    
    data_file = sys.argv[1]
    distance_file = sys.argv[2]
    
    solver = VRPSolver(data_file, distance_file)
    solution = solver.solve()
    
    print(f"총 차량 수: {len(solution['vehicles'])}")
    print(f"총 비용: {solution['total_cost']:,}원")
    print(f"실행 시간: {solution['execution_time']:.2f}초")
    
    solver.save_result(solution)

if __name__ == "__main__":
    main()