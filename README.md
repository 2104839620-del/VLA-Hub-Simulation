# VLA智能体交通枢纽仿真系统

基于Vision-Language-Action架构的综合交通枢纽多目标优化仿真系统。

## 📖 论文对应
本代码完整实现论文《综合交通枢纽的VLA多模态智能体设计与优化仿真》中的所有算法。

## 🚀 快速开始

### 环境要求
- MATLAB R2020a 或更高版本
- Statistics and Machine Learning Toolbox
- Mapping Toolbox（用于OSM底图显示）

### 运行步骤
1. 下载本仓库
2. 在MATLAB中打开 `VLA_Hub_Simulation_OSM_Refactored.m`
3. 直接运行主函数
4. 查看 `VLA_outputs_refactored` 目录中的结果

### 主要功能
- 🗺️ 真实OSM地理数据集成
- 🎯 双ε-constraint多目标优化
- 👥 社会力行人仿真模型
- 📊 自动化KPI评估与可视化

## 📁 文件结构
- `VLA_Hub_Simulation_OSM_Refactored.m` - 主程序
- `README.md` - 说明文档

## 🔧 核心算法
1. **候选生成**: DBSCAN + KMeans 融合聚类
2. **多目标优化**: LogSum可达性 + 成本约束 + 公平性
3. **双ε约束**: 站点数 × 成本上限网格搜索
4. **行人仿真**: 社会力模型

## 📊 输出结果
- Pareto前沿可视化
- OSM地图选址结果
- 行人分布仿真
- 详细KPI报告

## 🤝 引用
如使用本代码，请引用相关论文。
