from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import math
import random
import os
from datetime import datetime

app = Flask(__name__)


class TrafficHubSimulator:
    def __init__(self):
        self.params = {
            'center': [39.864444, 116.378558],
            'nDemand': 150,
            'nHubs': 12
        }

    def generate_synthetic_data(self, center, n_points):
        """生成合成数据"""
        lat, lon = center
        points = []
        for i in range(n_points):
            new_lat = lat + random.uniform(-0.01, 0.01)
            new_lon = lon + random.uniform(-0.01, 0.01)
            points.append({
                'lat': new_lat,
                'lon': new_lon,
                'name': f'POI_{i + 1}',
                'type': random.choice(['amenity', 'shop', 'transport'])
            })
        return points

    def generate_candidate_hubs(self, center, n_hubs):
        """生成候选枢纽"""
        lat, lon = center
        hubs = []
        for i in range(n_hubs):
            hub_lat = lat + random.uniform(-0.005, 0.005)
            hub_lon = lon + random.uniform(-0.005, 0.005)
            hubs.append({
                'id': i,
                'lat': hub_lat,
                'lon': hub_lon,
                'selected': i < min(4, n_hubs)
            })
        return hubs

    def calculate_kpi(self, selected_hubs):
        """计算KPI指标"""
        if not selected_hubs:
            return {
                'num_selected': 0,
                'coverage_30min': 0,
                'avg_travel_time': 0,
                'fairness': 0,
                'total_cost': 0,
                'mean_spacing': 0
            }

        return {
            'num_selected': len(selected_hubs),
            'coverage_30min': round(random.uniform(0.7, 0.9), 3),
            'avg_travel_time': round(random.uniform(15, 25), 1),
            'fairness': round(random.uniform(0.6, 0.9), 3),
            'total_cost': len(selected_hubs) * 4200,
            'mean_spacing': round(random.uniform(1.5, 3.0), 2)
        }

    def run_simulation(self, center, n_demand, n_hubs):
        """运行仿真"""
        try:
            print(f"开始仿真: 中心点{center}, 需求点{n_demand}, 枢纽{n_hubs}")

            # 生成数据
            poi_data = self.generate_synthetic_data(center, n_demand)
            hubs_data = self.generate_candidate_hubs(center, n_hubs)
            selected_hubs = [hub for hub in hubs_data if hub['selected']]

            # 计算KPI
            kpi = self.calculate_kpi(selected_hubs)

            return {
                'success': True,
                'kpi': kpi,
                'selected_hubs': len(selected_hubs),
                'total_candidates': len(hubs_data),
                'poi_count': len(poi_data),
                'hubs_data': hubs_data,
                'poi_data': poi_data,
                'center': center
            }

        except Exception as e:
            print(f"仿真错误: {e}")
            return {
                'success': False,
                'error': str(e)
            }


simulator = TrafficHubSimulator()


@app.route('/')
def index():
    """主页面 - 直接返回HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>交通枢纽仿真系统</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <style>
            #map { 
                height: 500px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                cursor: crosshair;
            }
            .card { margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .region-btn { margin: 2px; }
            .custom-coords { background: #f8f9fa; padding: 10px; border-radius: 5px; }
            .simulation-loading { display: none; text-align: center; padding: 10px; }
            .kpi-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .result-item { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .selected-marker { 
                background-color: #28a745; 
                border: 3px solid white; 
                border-radius: 50%; 
                width: 20px; 
                height: 20px; 
            }
            .click-hint {
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                z-index: 1000;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container-fluid mt-3">
            <div class="row">
                <div class="col-12">
                    <h2>🚗 交通枢纽智能体仿真系统</h2>
                    <p class="text-muted">支持地图点击选点功能</p>
                </div>
            </div>

            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5 class="card-title mb-0">📍 选择仿真区域</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label class="form-label">🏙️ 预设城市</label>
                                <div class="d-grid gap-2">
                                    <button class="btn btn-outline-primary region-btn" onclick="selectRegion('beijing')">北京南站</button>
                                    <button class="btn btn-outline-primary region-btn" onclick="selectRegion('shanghai')">上海虹桥</button>
                                    <button class="btn btn-outline-primary region-btn" onclick="selectRegion('guangzhou')">广州南站</button>
                                    <button class="btn btn-outline-primary region-btn" onclick="selectRegion('shenzhen')">深圳北站</button>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">🎯 自定义坐标</label>
                                <div class="custom-coords">
                                    <div class="row">
                                        <div class="col-6">
                                            <label class="form-label small">经度</label>
                                            <input type="number" class="form-control form-control-sm" id="customLng" step="0.0001" value="116.378558">
                                        </div>
                                        <div class="col-6">
                                            <label class="form-label small">纬度</label>
                                            <input type="number" class="form-control form-control-sm" id="customLat" step="0.0001" value="39.864444">
                                        </div>
                                    </div>
                                    <button class="btn btn-success btn-sm w-100 mt-2" onclick="useCustomCoords()">使用自定义坐标</button>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">🗺️ 地图选点</label>
                                <div class="alert alert-info small">
                                    💡 <strong>点击地图任意位置选择中心点</strong><br>
                                    点击后坐标会自动填入上方输入框
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">⚙️ 仿真参数</label>
                                <div class="row">
                                    <div class="col-6">
                                        <label class="form-label small">需求点数量</label>
                                        <input type="number" class="form-control form-control-sm" id="nDemand" value="150" min="50" max="1000">
                                    </div>
                                    <div class="col-6">
                                        <label class="form-label small">候选枢纽数</label>
                                        <input type="number" class="form-control form-control-sm" id="nHubs" value="12" min="3" max="30">
                                    </div>
                                </div>
                            </div>

                            <button class="btn btn-primary w-100" onclick="runSimulation()">🚀 开始智能体仿真</button>

                            <div class="simulation-loading mt-2" id="simulationLoading">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">加载中...</span>
                                </div>
                                <p class="mt-2">仿真计算中，请稍候...</p>
                            </div>

                            <div class="mt-3">
                                <button class="btn btn-outline-info btn-sm w-100" onclick="testConnection()">测试服务器连接</button>
                            </div>
                        </div>
                    </div>

                    <div class="card kpi-card">
                        <div class="card-header">
                            <h5 class="card-title mb-0 text-white">📊 仿真结果</h5>
                        </div>
                        <div class="card-body">
                            <div id="results">
                                <p class="text-center">等待仿真运行...</p>
                            </div>
                        </div>
                    </div>

                    <div class="card mt-3">
                        <div class="card-header bg-light">
                            <h5 class="card-title mb-0">📍 当前选择</h5>
                        </div>
                        <div class="card-body">
                            <div id="currentSelection">
                                <p class="text-muted small">尚未选择位置</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header bg-light d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">智能体仿真地图</h5>
                            <div>
                                <span class="badge bg-success" id="clickHint">点击地图选择位置</span>
                                <button class="btn btn-sm btn-outline-secondary" onclick="clearSelection()">清除选择</button>
                            </div>
                        </div>
                        <div class="card-body p-0 position-relative">
                            <div id="map"></div>
                        </div>
                    </div>

                    <div class="card mt-3">
                        <div class="card-header bg-light">
                            <h5 class="card-title mb-0">ℹ️ 系统状态</h5>
                        </div>
                        <div class="card-body">
                            <div id="status">
                                <p>系统就绪，请点击地图选择位置或使用预设城市</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script>
            // 预设城市坐标
            const regions = {
                'beijing': { name: '北京南站', lat: 39.864444, lng: 116.378558 },
                'shanghai': { name: '上海虹桥', lat: 31.193687, lng: 121.318542 },
                'guangzhou': { name: '广州南站', lat: 22.989383, lng: 113.270707 },
                'shenzhen': { name: '深圳北站', lat: 22.611362, lng: 114.029531 }
            };

            let currentMap = null;
            let selectedMarker = null;
            let currentCenter = [39.864444, 116.378558];

            // 初始化地图
            function initMap() {
                currentMap = L.map('map').setView(currentCenter, 14);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(currentMap);

                // 添加点击事件监听
                currentMap.on('click', function(e) {
                    selectPointOnMap(e.latlng.lat, e.latlng.lng);
                });

                updateStatus('地图初始化完成，点击地图选择位置');
            }

            // 地图点击选择位置
            function selectPointOnMap(lat, lng) {
                // 更新当前中心点
                currentCenter = [lat, lng];

                // 清除之前的标记
                if (selectedMarker) {
                    currentMap.removeLayer(selectedMarker);
                }

                // 添加新的标记
                selectedMarker = L.marker([lat, lng], {
                    icon: L.divIcon({
                        className: 'selected-marker',
                        html: '📍',
                        iconSize: [30, 30],
                        iconAnchor: [15, 15]
                    })
                }).addTo(currentMap)
                .bindPopup(`<b>选中位置</b><br>纬度: ${lat.toFixed(6)}<br>经度: ${lng.toFixed(6)}`)
                .openPopup();

                // 更新输入框
                document.getElementById('customLat').value = lat.toFixed(6);
                document.getElementById('customLng').value = lng.toFixed(6);

                // 更新显示
                updateCurrentSelection(lat, lng);
                updateStatus(`已选择位置: ${lat.toFixed(6)}, ${lng.toFixed(6)}`);

                // 移动地图视图到选中位置
                currentMap.setView([lat, lng], 14);
            }

            // 清除选择
            function clearSelection() {
                if (selectedMarker) {
                    currentMap.removeLayer(selectedMarker);
                    selectedMarker = null;
                }
                document.getElementById('customLat').value = '';
                document.getElementById('customLng').value = '';
                updateCurrentSelection(null, null);
                updateStatus('已清除选择');
            }

            // 更新当前选择显示
            function updateCurrentSelection(lat, lng) {
                const selectionDiv = document.getElementById('currentSelection');
                if (lat && lng) {
                    selectionDiv.innerHTML = `
                        <div class="alert alert-success py-2">
                            <strong>📍 已选择位置</strong><br>
                            <small>纬度: ${lat.toFixed(6)}<br>经度: ${lng.toFixed(6)}</small>
                        </div>
                    `;
                } else {
                    selectionDiv.innerHTML = '<p class="text-muted small">尚未选择位置</p>';
                }
            }

            // 选择预设区域
            function selectRegion(regionKey) {
                const region = regions[regionKey];
                if (region) {
                    selectPointOnMap(region.lat, region.lng);
                    updateStatus(`已选择预设区域: ${region.name}`);
                }
            }

            // 使用自定义坐标
            function useCustomCoords() {
                const lat = parseFloat(document.getElementById('customLat').value);
                const lng = parseFloat(document.getElementById('customLng').value);

                if (!isNaN(lat) && !isNaN(lng)) {
                    selectPointOnMap(lat, lng);
                } else {
                    alert('请输入有效的经纬度坐标！');
                }
            }

            // 测试服务器连接
            async function testConnection() {
                updateStatus('正在测试服务器连接...');
                try {
                    const response = await fetch('/test');
                    const data = await response.json();
                    updateStatus(`服务器连接正常: ${data.message}`);
                } catch (error) {
                    updateStatus(`服务器连接失败: ${error.message}`);
                    console.error('连接测试失败:', error);
                }
            }

            // 运行仿真
            async function runSimulation() {
                const lat = parseFloat(document.getElementById('customLat').value);
                const lng = parseFloat(document.getElementById('customLng').value);
                const nDemand = parseInt(document.getElementById('nDemand').value);
                const nHubs = parseInt(document.getElementById('nHubs').value);

                if (isNaN(lat) || isNaN(lng)) {
                    alert('请先选择有效的位置！点击地图或使用预设城市。');
                    return;
                }

                // 显示加载状态
                document.getElementById('simulationLoading').style.display = 'block';
                updateStatus('开始仿真计算...');

                try {
                    const response = await fetch('/run_simulation', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            center: [lat, lng],
                            nDemand: nDemand,
                            nHubs: nHubs
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP错误: ${response.status}`);
                    }

                    const result = await response.json();

                    if (result.success) {
                        // 在地图上显示仿真结果
                        displaySimulationResults(result);
                        updateStatus(`仿真完成! 生成${result.poi_count}个POI点，${result.total_candidates}个候选枢纽`);
                    } else {
                        updateStatus(`仿真失败: ${result.error}`);
                        alert(`仿真失败: ${result.error}`);
                    }
                } catch (error) {
                    console.error('请求失败:', error);
                    updateStatus(`请求失败: ${error.message}`);
                    alert(`请求失败: ${error.message}`);
                } finally {
                    document.getElementById('simulationLoading').style.display = 'none';
                }
            }

            // 在地图上显示仿真结果
            function displaySimulationResults(result) {
                // 清除之前的仿真结果（保留选择标记）
                currentMap.eachLayer((layer) => {
                    if (layer !== selectedMarker && layer instanceof L.TileLayer === false) {
                        currentMap.removeLayer(layer);
                    }
                });

                // 添加POI点
                if (result.poi_data) {
                    result.poi_data.forEach(poi => {
                        L.circleMarker([poi.lat, poi.lon], {
                            radius: 3,
                            color: 'blue',
                            fillColor: '#30f',
                            fillOpacity: 0.5
                        }).addTo(currentMap).bindPopup(`POI: ${poi.name}`);
                    });
                }

                // 添加候选枢纽
                if (result.hubs_data) {
                    result.hubs_data.forEach(hub => {
                        const isSelected = hub.selected;
                        const color = isSelected ? 'red' : 'orange';
                        const radius = isSelected ? 8 : 6;

                        L.circleMarker([hub.lat, hub.lon], {
                            radius: radius,
                            color: color,
                            fillColor: color,
                            fillOpacity: 0.7
                        }).addTo(currentMap).bindPopup(
                            isSelected ? 
                            `✅ 选中枢纽 ${hub.id + 1}` : 
                            `⭕ 候选枢纽 ${hub.id + 1}`
                        );

                        // 为选中的枢纽添加服务半径
                        if (isSelected) {
                            L.circle([hub.lat, hub.lon], {
                                color: 'green',
                                fillColor: 'green',
                                fillOpacity: 0.1,
                                radius: 500
                            }).addTo(currentMap).bindPopup('服务半径: 500米');
                        }
                    });
                }

                // 显示KPI结果
                displayKPIResults(result.kpi, result.selected_hubs, result.total_candidates);
            }

            // 显示KPI结果
            function displayKPIResults(kpi, selectedHubs, totalCandidates) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <strong>选中枢纽:</strong> ${selectedHubs} / ${totalCandidates}
                    </div>
                    <div class="result-item">
                        <strong>30分钟覆盖率:</strong> ${(kpi.coverage_30min * 100).toFixed(1)}%
                    </div>
                    <div class="result-item">
                        <strong>平均出行时间:</strong> ${kpi.avg_travel_time.toFixed(1)} 分钟
                    </div>
                    <div class="result-item">
                        <strong>公平性指数:</strong> ${kpi.fairness.toFixed(3)}
                    </div>
                    <div class="result-item">
                        <strong>总成本:</strong> ${kpi.total_cost.toFixed(0)}
                    </div>
                    <div class="result-item">
                        <strong>平均站间距:</strong> ${kpi.mean_spacing.toFixed(2)} km
                    </div>
                    <div class="mt-3 text-center">
                        <small class="text-white-50">仿真完成: ${new Date().toLocaleTimeString()}</small>
                    </div>
                `;
            }

            // 更新状态信息
            function updateStatus(message) {
                document.getElementById('status').innerHTML = `<p class="mb-0">${message}</p>`;
                console.log('状态:', message);
            }

            // 页面加载时初始化
            document.addEventListener('DOMContentLoaded', function() {
                initMap();
                updateStatus('系统就绪，请点击地图选择位置');
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """运行仿真"""
    try:
        data = request.get_json()
        print("收到请求数据:", data)

        center = data.get('center', [39.864444, 116.378558])
        n_demand = data.get('nDemand', 150)
        n_hubs = data.get('nHubs', 12)

        result = simulator.run_simulation(center, n_demand, n_hubs)
        return jsonify(result)

    except Exception as e:
        print("服务器错误:", e)
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        })


@app.route('/test')
def test():
    """测试接口"""
    return jsonify({'status': 'ok', 'message': '服务器运行正常'})


if __name__ == '__main__':
    print("启动交通枢纽仿真服务器...")
    print("访问地址: http://localhost:5000")
    print("功能说明:")
    print("1. 点击地图任意位置选择中心点")
    print("2. 或使用预设城市按钮")
    print("3. 设置仿真参数后点击开始仿真")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
