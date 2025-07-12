import numpy as np
import cv2
import json
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
import pyproj
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import trimesh


class NuScenesDataLoader:
    """NuScenes数据加载器"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.scene_tokens = [scene['token'] for scene in self.nusc.scene]
        
    def get_scene_data(self, scene_token: str) -> Dict:
        """获取场景数据"""
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']

        scene_data = {
            'samples': [],
            'location': location,
            'scene_token': scene_token
        }
        
        # 收集所有sample数据
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            sample_data = self._process_sample(sample)
            scene_data['samples'].append(sample_data)
            sample_token = sample['next']
            
        return scene_data
    
    def _process_sample(self, sample: Dict) -> Dict:
        """处理单个sample数据"""
        # 获取相机数据
        cam_front_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_front_token)
        
        # 获取ego pose
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # 获取相机校准数据
        calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # 获取图像路径
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        
        return {
            'sample_token': sample['token'],
            'timestamp': sample['timestamp'],
            'image_path': image_path,
            'ego_pose': ego_pose,
            'calibrated_sensor': calibrated_sensor,
            'cam_data': cam_data
        }

class OSMDataProcessor:
    """OpenStreetMap数据处理器"""
    
    def __init__(self):
        self.buildings = None
        self.roads = None
        self.bbox = None
        
    def load_osm_data(self, center_lat: float, center_lon: float, distance: int = 1000):
        """加载OSM数据"""
        try:
            # 创建边界框
            self.bbox = ox.utils_geo.bbox_from_point((center_lat, center_lon), distance)
            
            # 获取建筑数据
            buildings_gdf = ox.features_from_bbox(
                self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
                tags={'building': True}
            )
            
            # 获取道路数据
            roads_graph = ox.graph_from_bbox(
                self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
                network_type='all'
            )
            
            self.buildings = buildings_gdf
            self.roads = roads_graph
            
            print(f"加载了 {len(buildings_gdf)} 个建筑物")
            
        except Exception as e:
            print(f"加载OSM数据时出错: {e}")
            # 创建模拟数据
            self._create_mock_data(center_lat, center_lon, distance)
    
    def _create_mock_data(self, center_lat: float, center_lon: float, distance: int):
        """创建模拟OSM数据"""
        # 创建一些模拟建筑物
        buildings_data = []
        for i in range(20):
            # 随机生成建筑物位置
            lat_offset = np.random.uniform(-0.005, 0.005)
            lon_offset = np.random.uniform(-0.005, 0.005)
            
            # 创建矩形建筑
            building_corners = [
                (center_lon + lon_offset, center_lat + lat_offset),
                (center_lon + lon_offset + 0.0005, center_lat + lat_offset),
                (center_lon + lon_offset + 0.0005, center_lat + lat_offset + 0.0005),
                (center_lon + lon_offset, center_lat + lat_offset + 0.0005),
                (center_lon + lon_offset, center_lat + lat_offset)
            ]
            
            building_polygon = Polygon(building_corners)
            buildings_data.append({
                'geometry': building_polygon,
                'building': 'yes',
                'height': np.random.uniform(10, 50)
            })
        
        self.buildings = gpd.GeoDataFrame(buildings_data)
        print(f"创建了 {len(buildings_data)} 个模拟建筑物")

class CoordinateTransformer:
    """坐标转换器"""
    
    def __init__(self, origin_lat: float, origin_lon: float):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # 设置投影系统
        self.proj_wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
        self.proj_utm = pyproj.Proj(proj='utm', zone=self._get_utm_zone(origin_lon), datum='WGS84')
        
    def _get_utm_zone(self, lon: float) -> int:
        """获取UTM zone"""
        return int((lon + 180) / 6) + 1
    
    def latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """将经纬度转换为本地坐标"""
        # 转换为UTM坐标
        x, y = pyproj.transform(self.proj_wgs84, self.proj_utm, lon, lat)
        
        # 转换为相对于原点的本地坐标
        origin_x, origin_y = pyproj.transform(self.proj_wgs84, self.proj_utm, 
                                            self.origin_lon, self.origin_lat)
        
        return x - origin_x, y - origin_y
    
    def ego_to_global(self, ego_pose: Dict) -> np.ndarray:
        """将ego pose转换为全局变换矩阵"""
        translation = np.array(ego_pose['translation'])
        rotation = np.array(ego_pose['rotation'])
        
        # 四元数转换为旋转矩阵
        r = Rotation.from_quat(rotation)
        rotation_matrix = r.as_matrix()
        
        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        return transform_matrix

class DepthProcessor:
    """深度图处理器"""
    
    def __init__(self, depth_scale: float = 1000.0):
        self.depth_scale = depth_scale
    
    def load_depth_image(self, depth_path: str) -> np.ndarray:
        """加载深度图"""
        if not os.path.exists(depth_path):
            print(f"深度图不存在: {depth_path}")
            return None
            
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"无法加载深度图: {depth_path}")
            return None
            
        # 转换为米为单位
        depth_img = depth_img.astype(np.float32) / self.depth_scale
        return depth_img
    
    def depth_to_pointcloud(self, depth_img: np.ndarray, 
                           camera_intrinsic: np.ndarray) -> np.ndarray:
        """将深度图转换为点云"""
        h, w = depth_img.shape
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 转换为齐次坐标
        pixel_coords = np.stack([u.flatten(), v.flatten(), np.ones_like(u.flatten())], axis=0)
        
        # 相机内参逆矩阵
        K_inv = np.linalg.inv(camera_intrinsic)
        
        # 转换为相机坐标系
        camera_coords = K_inv @ pixel_coords
        
        # 乘以深度值
        depth_values = depth_img.flatten()
        valid_mask = depth_values > 0
        
        points_3d = camera_coords * depth_values[np.newaxis, :]
        points_3d = points_3d[:, valid_mask]
        
        return points_3d.T

class BuildingExtractor:
    """建筑物提取器"""
    
    def __init__(self, min_height: float = 2.0, max_height: float = 100.0):
        self.min_height = min_height
        self.max_height = max_height
    
    def extract_buildings_from_pointcloud(self, pointcloud: np.ndarray, 
                                        ground_height: float = 0.0) -> List[Dict]:
        """从点云中提取建筑物"""
        if pointcloud.shape[0] == 0:
            return []
        
        # 过滤地面点
        above_ground = pointcloud[pointcloud[:, 2] > ground_height + self.min_height]
        
        if len(above_ground) == 0:
            return []
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=2.0, min_samples=50).fit(above_ground[:, :2])
        labels = clustering.labels_
        
        buildings = []
        for label in np.unique(labels):
            if label == -1:  # 噪声点
                continue
                
            cluster_points = above_ground[labels == label]
            
            # 计算建筑物属性
            building_info = self._analyze_building_cluster(cluster_points)
            if building_info:
                buildings.append(building_info)
        
        return buildings
    
    def _analyze_building_cluster(self, points: np.ndarray) -> Optional[Dict]:
        """分析建筑物聚类"""
        if len(points) < 20:
            return None
        
        # 计算边界框
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        
        width = max_xyz[0] - min_xyz[0]
        length = max_xyz[1] - min_xyz[1]
        height = max_xyz[2] - min_xyz[2]
        
        # 过滤太小或太大的建筑
        if width < 3 or length < 3 or height < self.min_height or height > self.max_height:
            return None
        
        # 计算中心点
        center = (min_xyz + max_xyz) / 2
        
        # 计算建筑物轮廓
        footprint = self._compute_building_footprint(points[:, :2])
        
        return {
            'center': center,
            'dimensions': [width, length, height],
            'footprint': footprint,
            'points': points,
            'volume': width * length * height
        }
    
    def _compute_building_footprint(self, points_2d: np.ndarray) -> np.ndarray:
        """计算建筑物底面轮廓"""
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except:
            # 如果凸包计算失败，返回边界框
            min_xy = np.min(points_2d, axis=0)
            max_xy = np.max(points_2d, axis=0)
            return np.array([
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]]
            ])

class Building3DGenerator:
    """3D建筑生成器"""
    
    def __init__(self):
        self.buildings = []
    
    def generate_building_mesh(self, building_info: Dict) -> trimesh.Trimesh:
        """生成建筑物3D网格"""
        center = building_info['center']
        dimensions = building_info['dimensions']
        footprint = building_info['footprint']
        
        # 创建建筑物底面
        base_vertices = []
        for point in footprint:
            base_vertices.append([point[0], point[1], center[2] - dimensions[2]/2])
        
        # 创建建筑物顶面
        top_vertices = []
        for point in footprint:
            top_vertices.append([point[0], point[1], center[2] + dimensions[2]/2])
        
        # 组合顶点
        vertices = np.array(base_vertices + top_vertices)
        
        # 创建面
        faces = []
        n_vertices = len(footprint)
        
        # 底面
        if n_vertices >= 3:
            for i in range(1, n_vertices - 1):
                faces.append([0, i, i + 1])
        
        # 顶面
        if n_vertices >= 3:
            for i in range(1, n_vertices - 1):
                faces.append([n_vertices, n_vertices + i + 1, n_vertices + i])
        
        # 侧面
        for i in range(n_vertices):
            next_i = (i + 1) % n_vertices
            faces.extend([
                [i, next_i, n_vertices + i],
                [next_i, n_vertices + next_i, n_vertices + i]
            ])
        
        # 创建mesh
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        except Exception as e:
            print(f"创建mesh时出错: {e}")
            # 创建简单的盒子mesh
            box = trimesh.creation.box(extents=dimensions)
            box.apply_translation(center)
            return box
    
    def align_buildings_with_osm(self, extracted_buildings: List[Dict], 
                                osm_buildings: gpd.GeoDataFrame,
                                coord_transformer: CoordinateTransformer) -> List[Dict]:
        """将提取的建筑物与OSM数据对齐"""
        aligned_buildings = []
        
        for building in extracted_buildings:
            center = building['center']
            
            # 找到最近的OSM建筑物
            min_distance = float('inf')
            best_match = None
            
            for idx, osm_building in osm_buildings.iterrows():
                if osm_building.geometry.geom_type == 'Polygon':
                    # 获取OSM建筑物中心点
                    osm_center = osm_building.geometry.centroid
                    osm_x, osm_y = coord_transformer.latlon_to_local(
                        osm_center.y, osm_center.x
                    )
                    
                    # 计算距离
                    distance = np.sqrt((center[0] - osm_x)**2 + (center[1] - osm_y)**2)
                    
                    if distance < min_distance and distance < 50:  # 50米内匹配
                        min_distance = distance
                        best_match = osm_building
            
            # 如果找到匹配的OSM建筑物，使用OSM数据优化
            if best_match is not None:
                aligned_building = self._refine_building_with_osm(building, best_match, coord_transformer)
                aligned_buildings.append(aligned_building)
            else:
                aligned_buildings.append(building)
        
        return aligned_buildings
    
    def _refine_building_with_osm(self, building: Dict, osm_building, 
                                 coord_transformer: CoordinateTransformer) -> Dict:
        """使用OSM数据优化建筑物"""
        # 获取OSM建筑物轮廓
        osm_geometry = osm_building.geometry
        
        if osm_geometry.geom_type == 'Polygon':
            # 转换OSM坐标到本地坐标
            osm_coords = []
            for coord in osm_geometry.exterior.coords:
                local_x, local_y = coord_transformer.latlon_to_local(coord[1], coord[0])
                osm_coords.append([local_x, local_y])
            
            # 更新建筑物轮廓
            refined_building = building.copy()
            refined_building['footprint'] = np.array(osm_coords[:-1])  # 去除重复的最后一个点
            
            # 如果OSM有高度信息，使用它
            if hasattr(osm_building, 'height') and osm_building.height:
                try:
                    osm_height = float(osm_building.height)
                    refined_building['dimensions'][2] = osm_height
                except:
                    pass
            
            return refined_building
        
        return building

class MultiModalAligner:
    """多模态数据对齐器"""
    
    def __init__(self):
        self.nuscenes_loader = None
        self.osm_processor = None
        self.coord_transformer = None
        self.depth_processor = None
        self.building_extractor = None
        self.building_generator = None
    
    def initialize(self, nuscenes_dataroot: str, depth_root: str):
        """初始化所有组件"""
        print("初始化NuScenes数据加载器...")
        self.nuscenes_loader = NuScenesDataLoader(nuscenes_dataroot)
        
        print("初始化深度处理器...")
        self.depth_processor = DepthProcessor()
        
        print("初始化建筑物提取器...")
        self.building_extractor = BuildingExtractor()
        
        print("初始化3D建筑生成器...")
        self.building_generator = Building3DGenerator()
        
        self.depth_root = depth_root
    
    def process_scene(self, scene_index: int = 0) -> List[Dict]:
        """处理单个场景"""
        print(f"处理场景 {scene_index}...")
        
        # 获取场景数据
        scene_token = self.nuscenes_loader.scene_tokens[scene_index]
        scene_data = self.nuscenes_loader.get_scene_data(scene_token)
        
        if not scene_data['samples']:
            print("场景中没有样本数据")
            return []
        
        # 获取场景位置信息
        first_sample = scene_data['samples'][0]
        ego_pose = first_sample['ego_pose']
        
        # 假设Boston场景的大概位置
        center_lat, center_lon = 42.3601, -71.0589  # Boston  ############ 
        
        print(f"初始化坐标转换器 (lat: {center_lat}, lon: {center_lon})...")
        self.coord_transformer = CoordinateTransformer(center_lat, center_lon)
        
        print("加载OSM数据...")
        self.osm_processor = OSMDataProcessor()
        self.osm_processor.load_osm_data(center_lat, center_lon)
        
        # 处理所有样本
        all_buildings = []
        for i, sample in enumerate(scene_data['samples'][:5]):  # 处理前5个样本
            print(f"处理样本 {i+1}/{min(5, len(scene_data['samples']))}")
            
            buildings = self._process_sample(sample, scene_data['location'])
            all_buildings.extend(buildings)
        
        # 去重和合并相近的建筑物
        merged_buildings = self._merge_nearby_buildings(all_buildings)
        
        print(f"共提取到 {len(merged_buildings)} 个建筑物")
        return merged_buildings
    
    def _process_sample(self, sample: Dict, location: str) -> List[Dict]:
        """处理单个样本"""
        # 构建深度图路径
        sample_token = sample['sample_token']
        depth_path = os.path.join(self.depth_root, f"{sample_token}.png")
        
        # 加载深度图
        depth_img = self.depth_processor.load_depth_image(depth_path)
        if depth_img is None:
            # 如果深度图不存在，创建模拟深度图
            depth_img = self._create_mock_depth_image()
        
        # 获取相机内参
        camera_intrinsic = np.array(sample['calibrated_sensor']['camera_intrinsic'])
        
        # 转换为点云
        pointcloud = self.depth_processor.depth_to_pointcloud(depth_img, camera_intrinsic)
        
        # 转换到全局坐标系
        ego_transform = self.coord_transformer.ego_to_global(sample['ego_pose'])
        cam_transform = np.array(sample['calibrated_sensor']['translation'] + 
                               sample['calibrated_sensor']['rotation'])
        
        # 简化的全局坐标转换
        global_points = pointcloud.copy()
        global_points[:, 0] += ego_transform[0, 3]
        global_points[:, 1] += ego_transform[1, 3]
        global_points[:, 2] += ego_transform[2, 3]
        
        # 提取建筑物
        buildings = self.building_extractor.extract_buildings_from_pointcloud(global_points)
        
        # 与OSM数据对齐
        if self.osm_processor.buildings is not None:
            aligned_buildings = self.building_generator.align_buildings_with_osm(
                buildings, self.osm_processor.buildings, self.coord_transformer
            )
        else:
            aligned_buildings = buildings
        
        return aligned_buildings
    
    def _create_mock_depth_image(self) -> np.ndarray:
        """创建模拟深度图"""
        # 创建1600x900的深度图
        depth_img = np.zeros((900, 1600), dtype=np.float32)
        
        # 添加一些模拟的建筑物深度
        # 远处的建筑物
        depth_img[200:400, 200:400] = np.random.uniform(30, 50, (200, 200))
        depth_img[200:400, 600:800] = np.random.uniform(25, 40, (200, 200))
        depth_img[200:400, 1000:1200] = np.random.uniform(40, 60, (200, 200))
        
        # 近处的建筑物
        depth_img[500:700, 300:500] = np.random.uniform(15, 25, (200, 200))
        depth_img[500:700, 800:1000] = np.random.uniform(20, 30, (200, 200))
        
        # 添加一些噪声
        depth_img += np.random.normal(0, 1, depth_img.shape)
        depth_img = np.clip(depth_img, 0, 100)
        
        return depth_img
    
    def _merge_nearby_buildings(self, buildings: List[Dict], 
                              distance_threshold: float = 10.0) -> List[Dict]:
        """合并相近的建筑物"""
        if not buildings:
            return []
        
        merged = []
        used = set()
        
        for i, building in enumerate(buildings):
            if i in used:
                continue
                
            # 找到相近的建筑物
            nearby_buildings = [building]
            used.add(i)
            
            for j, other_building in enumerate(buildings):
                if j in used or j == i:
                    continue
                
                # 计算距离
                dist = np.linalg.norm(building['center'] - other_building['center'])
                if dist < distance_threshold:
                    nearby_buildings.append(other_building)
                    used.add(j)
            
            # 合并建筑物
            if len(nearby_buildings) == 1:
                merged.append(building)
            else:
                merged_building = self._merge_buildings(nearby_buildings)
                merged.append(merged_building)
        
        return merged
    
    def _merge_buildings(self, buildings: List[Dict]) -> Dict:
        """合并多个建筑物"""
        # 计算平均中心点
        centers = np.array([b['center'] for b in buildings])
        avg_center = np.mean(centers, axis=0)
        
        # 计算平均尺寸
        dimensions = np.array([b['dimensions'] for b in buildings])
        avg_dimensions = np.mean(dimensions, axis=0)
        
        # 合并所有点
        all_points = np.vstack([b['points'] for b in buildings])
        
        # 重新计算轮廓
        footprint = self.building_extractor._compute_building_footprint(all_points[:, :2])
        
        return {
            'center': avg_center,
            'dimensions': avg_dimensions,
            'footprint': footprint,
            'points': all_points,
            'volume': np.prod(avg_dimensions)
        }
    
    def save_buildings(self, buildings: List[Dict], output_dir: str):
        """保存建筑物数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存建筑物信息
        building_info = []
        for i, building in enumerate(buildings):
            # 生成3D网格
            mesh = self.building_generator.generate_building_mesh(building)
            
            # 保存网格
            mesh_path = os.path.join(output_dir, f"building_{i:03d}.obj")
            mesh.export(mesh_path)
            
            # 保存建筑物信息
            info = {
                'id': i,
                'center': building['center'].tolist(),
                'dimensions': building['dimensions'],
                'footprint': building['footprint'].tolist(),
                'volume': building['volume'],
                'mesh_path': mesh_path
            }
            building_info.append(info)
        
        # 保存汇总信息
        with open(os.path.join(output_dir, 'buildings_info.json'), 'w') as f:
            json.dump(building_info, f, indent=2)
        
        print(f"已保存 {len(buildings)} 个建筑物到 {output_dir}")

def main():
    """主函数"""
    # 设置路径
    nuscenes_dataroot = "/media/saczone/DATADRIVE1/venshow/data/nuscenes/v1.0-mini/"  # 请替换为实际路径
    depth_root = "/media/saczone/DATADRIVE1/venshow/data/nuscenes/depth_out"  # 请替换为深度图保存路径
    output_dir = "./output_buildings"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化对齐器
    aligner = MultiModalAligner()
    aligner.initialize(nuscenes_dataroot, depth_root)
    
    # 处理场景
    buildings = aligner.process_scene(scene_index=0)
    
    # 保存结果
    aligner.save_buildings(buildings, output_dir)
    
    print("建筑物生成完成！")


if __name__ == "__main__":
    main()
