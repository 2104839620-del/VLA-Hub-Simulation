function VLA_Hub_Simulation_OSM_Refactored()
%% ==========================================================
%   VLA 智能体交通枢纽仿真 - 重构优化版
%   - 修复目标函数方向问题
%   - 向量化距离计算
%   - 严格的约束处理
%   - 集中参数管理
% ==========================================================
clc; clear; close all; rng(42);

%% ---------------- 0) 集中参数配置 ----------------
params = struct();
params.center = [39.864444, 116.378558];   % 北京南站
params.radius_m = 1500;
params.nDemand = 1500;
params.nAgents = 600;
params.simulation_steps = 120;

% 算法参数
params.UNIT_COST = 4200;
params.BETA_LOGSUM = 0.18;
params.DISP_WEIGHT = 0.35;
params.USE_GINI = false;
params.useKMeansCandidates = true;
params.targetK = [];

% 策略参数
policy.rules.min_service_radius_m = 500;
policy.rules.max_noise_db = 65;
policy.rules.max_sites = 4;
policy.rules.min_coverage = 0.75;  % 强化覆盖率
policy.rules.objective_weights = [1.0, 0.35, 0.8];

% GA参数
params.ga_options = optimoptions('ga', ...
    'PopulationSize', 160, ...
    'MaxGenerations', 140, ...
    'Display', 'off', ...
    'UseVectorized', false, ...
    'FunctionTolerance', 1e-6);

% 输出目录 - 定义为全局可访问的变量
global outdir;  % 添加这一行
outdir = fullfile(pwd,'VLA_outputs_refactored');
if ~exist(outdir,'dir'), mkdir(outdir); end

%% ---------------- 1) OSM POI 获取（保持原逻辑） ----------------
disp('=== [1] 获取 OSM POI 数据 ===');
[poi_table, use_real_poi, geo_bounds] = fetch_and_prepare_poi_data(params);

%% ---------------- 2) 候选生成（DBSCAN + KMeans） ----------------
disp('=== [2] 候选枢纽生成 ===');
[cand, cand_lat, cand_lon, nCand] = generate_candidate_hubs(poi_table, use_real_poi, geo_bounds, params);

%% ---------------- 3) 需求点生成 ----------------
disp('=== [3] 需求点生成 ===');
[demand_xy, demand_lat, demand_lon] = generate_demand_points(poi_table, use_real_poi, geo_bounds, cand, params);

%% ---------------- 4) 向量化出行时间计算 ----------------
disp('=== [4] 计算出行时间矩阵 ===');
travelTime = compute_travel_time_matrix(cand_lat, cand_lon, demand_lat, demand_lon);

%% ---------------- 5) 双 ε-约束 Pareto 搜索 ----------------
disp('=== [5] 双 ε-约束 Pareto 搜索 ===');
[Xpareto, Fpareto, x_best, selected] = run_epsilon_constraint_search(...
    nCand, travelTime, policy.rules, params, cand);

%% ---------------- 6) 结果可视化 ----------------
disp('=== [6] 结果可视化 ===');
visualize_results(poi_table, cand_lat, cand_lon, selected, geo_bounds, policy.rules);

%% ---------------- 7) 行人仿真与KPI评估 ----------------
disp('=== [7] 行人仿真与KPI评估 ===');
run_pedestrian_simulation_and_kpi(selected, cand, travelTime, params, policy.rules);

disp('=== 重构完成：所有结果已保存 ===');
end

%% ===================== 核心重构函数 =====================

function [poi_table, use_real_poi, geo_bounds] = fetch_and_prepare_poi_data(params)
% 获取并准备POI数据
global outdir;  % 添加这一行

use_real_poi = false;
poi_table = table();
geo_bounds = struct();

try
    poi_table = fetch_osm_poi_refactored(params.center, params.radius_m);
    writetable(poi_table, fullfile(outdir,'poi_osm.csv'));
    
    % 记录数据来源
    fid = fopen(fullfile(outdir,'sources.txt'),'w');
    fprintf(fid, 'Data Source: OpenStreetMap via Overpass API\n');
    fprintf(fid, 'Refactored Version: Fixed algorithm issues\n');
    fclose(fid);
    
    use_real_poi = height(poi_table) >= 50;
    disp(['OSM 拉取成功，POI 数量: ', num2str(height(poi_table))]);
    
    % 计算地理边界
    geo_bounds.minlat = min(poi_table.lat);
    geo_bounds.maxlat = max(poi_table.lat);
    geo_bounds.minlon = min(poi_table.lon);
    geo_bounds.maxlon = max(poi_table.lon);
    
catch ME
    warning('OSM 拉取失败，使用合成 POI: %s', ME.message);
    
    % 合成数据的地理边界
    geo_bounds.minlat = params.center(1) - 0.02;
    geo_bounds.maxlat = params.center(1) + 0.02;
    geo_bounds.minlon = params.center(2) - 0.02;
    geo_bounds.maxlon = params.center(2) + 0.02;
    
    % 生成合成POI
    nPOI = 800;
    poi_norm = generate_synthetic_POI(nPOI);
    lat = geo_bounds.minlat + poi_norm(:,2)*(geo_bounds.maxlat - geo_bounds.minlat);
    lon = geo_bounds.minlon + poi_norm(:,1)*(geo_bounds.maxlon - geo_bounds.minlon);
    
    poi_table = table(lat, lon, repmat({''}, size(lat)), repmat({''}, size(lat)), ...
        'VariableNames', {'lat', 'lon', 'name', 'tags'});
end
end

function [cand, cand_lat, cand_lon, nCand] = generate_candidate_hubs(poi_table, use_real_poi, geo_bounds, params)
% 生成候选枢纽位置
global outdir;  % 添加这一行

% 归一化POI坐标
if use_real_poi
    lat = poi_table.lat; lon = poi_table.lon;
    poi_norm = [(lon - geo_bounds.minlon)./(geo_bounds.maxlon - geo_bounds.minlon + eps), ...
                (lat - geo_bounds.minlat)./(geo_bounds.maxlat - geo_bounds.minlat + eps)];
else
    nPOI = 800;
    poi_norm = generate_synthetic_POI(nPOI);
end

% DBSCAN聚类
k = 5;
D = pdist2(poi_norm, poi_norm);
Dsorted = sort(D + diag(inf(size(D,1),1)), 2);
kdist = Dsorted(:,k);
eps_val = median(kdist) * 1.1;
MinPts = max(8, round(size(poi_norm,1)/100));
idx = dbscan(poi_norm, eps_val, MinPts);

valid = idx > 0;
labels = unique(idx(valid));
cand_db = zeros(numel(labels), 2);
for kk = 1:numel(labels)
    pts = poi_norm(idx == labels(kk), :);
    cand_db(kk, :) = mean(pts, 1);
end

% KMeans补充
cand = cand_db;
if params.useKMeansCandidates
    if isempty(params.targetK)
        targetK = max(4, min(12, round(size(poi_norm,1)/300)));
    else
        targetK = params.targetK;
    end
    opts = statset('MaxIter', 300, 'UseParallel', false);
    [~, Ck] = kmeans(poi_norm, targetK, 'Replicates', 5, 'Distance', 'sqeuclidean', 'Options', opts);
    cand = unique([cand; Ck], 'rows', 'stable');
end

cand = min(max(cand, 0), 1);
nCand = size(cand, 1);

% 转换为实际经纬度
cand_lon = cand(:,1)*(geo_bounds.maxlon - geo_bounds.minlon) + geo_bounds.minlon;
cand_lat = cand(:,2)*(geo_bounds.maxlat - geo_bounds.minlat) + geo_bounds.minlat;

% 可视化
figure('Name', 'POI与候选地块', 'Position', [100 100 800 600]);
scatter(poi_norm(:,1), poi_norm(:,2), 10, [0.6 0.8 1], 'filled'); hold on;
scatter(cand(:,1), cand(:,2), 80, 'r', 'filled');
title(sprintf('候选交通枢纽（DBSCAN %d + KMEANS）', size(cand_db,1)));
xlabel('X (归一化)'); ylabel('Y (归一化)'); 
legend('POI', '候选枢纽', 'Location', 'best'); grid on;
saveas(gcf, fullfile(outdir,'candidate_generation.png'));
end

function [demand_xy, demand_lat, demand_lon] = generate_demand_points(poi_table, use_real_poi, geo_bounds, cand, params)
% 生成需求点

if use_real_poi
    [demand_lat, demand_lon] = sample_kde_points_refactored(poi_table.lat, poi_table.lon, params.nDemand);
    demand_lat = min(max(demand_lat, geo_bounds.minlat), geo_bounds.maxlat);
    demand_lon = min(max(demand_lon, geo_bounds.minlon), geo_bounds.maxlon);
else
    [demand_lat, demand_lon] = sample_multimodal_hotspots_refactored(...
        cand, params.nDemand, geo_bounds);
end

demand_xy = [(demand_lon - geo_bounds.minlon)./(geo_bounds.maxlon - geo_bounds.minlon + eps), ...
             (demand_lat - geo_bounds.minlat)./(geo_bounds.maxlat - geo_bounds.minlat + eps)];
end

function travelTime = compute_travel_time_matrix(cand_lat, cand_lon, demand_lat, demand_lon)
% 向量化计算出行时间矩阵
global outdir;  % 添加这一行

nCand = length(cand_lat);
nDemand = length(demand_lat);

% 一次性计算所有点对距离
fprintf('计算 %d×%d 出行时间矩阵...\n', nCand, nDemand);

% 使用meshgrid向量化计算
[LAT_CAND, LAT_DEMAND] = meshgrid(cand_lat, demand_lat);
[LON_CAND, LON_DEMAND] = meshgrid(cand_lon, demand_lon);

% 向量化Haversine计算
dist_km = haversine_vectorized(LAT_CAND, LON_CAND, LAT_DEMAND, LON_DEMAND);

% 转换为时间（分钟）：基础时间 + 距离时间 + 随机扰动
travelTime = 2 + 8 * dist_km' + 3 * randn(nCand, nDemand);
travelTime = max(travelTime, 1);  % 最小1分钟

writematrix(travelTime, fullfile(outdir,'travelTime_matrix.csv'));
disp('出行时间矩阵计算完成');
end

function [Xpareto, Fpareto, x_best, selected] = run_epsilon_constraint_search(...
    nCand, travelTime, rules, params, cand)
% 运行双ε约束Pareto搜索
global outdir;  % 添加这一行

COST_CAP_GRID = linspace(params.UNIT_COST*1, params.UNIT_COST*rules.max_sites*1.6, 7);
allX = []; allF = [];

fprintf('开始双ε约束搜索 (%d 候选点)...\n', nCand);

for kAllow = 1:rules.max_sites
    for cap = COST_CAP_GRID
        fprintf('搜索: k=%d, cost_cap=%.0f\n', kAllow, cap);
        
        if nCand <= 26
            % 枚举搜索
            X = enumerate_exact_k(nCand, kAllow);
            F = evaluate_population_refactored(X, travelTime, rules, params, cand, cap);
            
            if ~isempty(F)
                [rep_idx, rep_F] = select_representative_solution(F);
                allX = [allX; X(rep_idx, :)]; %#ok<AGROW>
                allF = [allF; rep_F]; %#ok<AGROW>
            end
        else
            % GA搜索
            nvars = nCand;
            fitnessFcn = @(x)evaluate_single_solution_refactored(...
                x, travelTime, rules, params, cand, kAllow, cap);
            
            [xBest, fBest] = ga(fitnessFcn, nvars, [], [], [], [], ...
                zeros(1,nvars), ones(1,nvars), [], 1:nvars, params.ga_options);
            
            if all(isfinite(fBest))
                allX = [allX; round(xBest) > 0]; %#ok<AGROW>
                allF = [allF; fBest]; %#ok<AGROW>
            end
        end
    end
end

% 去重和Pareto过滤
if ~isempty(allX)
    [allX, ia] = unique(allX, 'rows', 'stable');
    allF = allF(ia, :);
    keep = is_pareto_efficient_refactored(allF);
    Xpareto = allX(keep, :);
    Fpareto = allF(keep, :);
else
    error('未找到可行解！请检查约束条件。');
end

% 选择最终解（理想点法）
[best_idx, selection_metrics] = select_best_solution(Fpareto, rules.objective_weights);
x_best = Xpareto(best_idx, :);
selected = find(x_best > 0);

% 保存结果
save_pareto_results(Xpareto, Fpareto, selected, cand, selection_metrics);
fprintf('Pareto搜索完成: %d个非支配解，选择%d个枢纽\n', size(Xpareto,1), length(selected));
end

function visualize_results(poi_table, cand_lat, cand_lon, selected, geo_bounds, rules)
% 可视化结果
global outdir;  % 添加这一行

try
    try_osm_basemap();
    f = figure('Name', 'OSM地图可视化', 'Position', [200 100 900 700]);
    ax = geoaxes; hold(ax, 'on'); 
    geobasemap(ax, 'osm');
    
    % POI
    geoscatter(ax, poi_table.lat, poi_table.lon, 6, 'filled', ...
        'MarkerFaceAlpha', 0.25, 'DisplayName', 'POI');
    
    % 候选点
    geoscatter(ax, cand_lat, cand_lon, 60, 'r', 'filled', ...
        'DisplayName', '候选点');
    
    % 选中点
    if ~isempty(selected)
        geoscatter(ax, cand_lat(selected), cand_lon(selected), 120, 'k', 'filled', ...
            'Marker', 'p', 'DisplayName', '选中枢纽');
        
        % 服务半径圈
        radius_deg = (rules.min_service_radius_m / 1000) / 111;
        for jj = 1:length(selected)
            th = linspace(0, 2*pi, 100);
            lat_circle = cand_lat(selected(jj)) + radius_deg * cos(th);
            lon_circle = cand_lon(selected(jj)) + radius_deg * sin(th) ./ cosd(cand_lat(selected(jj)));
            geoplot(ax, lat_circle, lon_circle, '-', 'LineWidth', 1.5, ...
                'Color', [0.1 0.5 1], 'DisplayName', sprintf('服务半径%d米', rules.min_service_radius_m));
            
            % 编号
            text(ax, cand_lat(selected(jj)), cand_lon(selected(jj)), ...
                sprintf('  枢纽%d', jj), 'FontSize', 11, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], 'BackgroundColor', 'white');
        end
    end
    
    geolimits(ax, [geo_bounds.minlat, geo_bounds.maxlat], ...
        [geo_bounds.minlon, geo_bounds.maxlon]);
    title(ax, '北京南站区域 - 交通枢纽选址优化结果 (OSM底图)');
    
    % 图例处理
    legend_labels = {'POI', '候选点'};
    if ~isempty(selected)
        legend_labels = [legend_labels, {'选中枢纽', '服务半径'}];
    end
    lgd = legend(ax, legend_labels, 'Location', 'northeastoutside');
    set(lgd, 'Interpreter', 'none');
    
    saveas(f, fullfile(outdir, 'final_map_visualization.png'));
    
catch ME
    warning('OSM底图失败，使用回退方案: %s', ME.message);
    create_fallback_visualization(poi_table, cand_lat, cand_lon, selected, geo_bounds);
end
end

function run_pedestrian_simulation_and_kpi(selected, cand, travelTime, params, rules)
% 行人仿真和KPI评估
global outdir;  % 添加这一行

if isempty(selected)
    warning('没有选中的枢纽，跳过行人仿真');
    return;
end

disp('运行行人社会力仿真...');
[final_pos, traj] = pedestrian_abm_refactored(selected, cand, params.nAgents, params.simulation_steps);

% 可视化行人分布
figure('Name', '行人分布仿真', 'Position', [200 200 800 650]);
scatter(final_pos(:,1), final_pos(:,2), 10, [0.2 0.7 0.2], 'filled'); hold on;
scatter(cand(selected,1), cand(selected,2), 150, 'r', 'filled', 'MarkerEdgeColor', 'k');
title('行人最终分布与枢纽选址'); 
xlabel('X (归一化)'); ylabel('Y (归一化)'); grid on;
saveas(gcf, fullfile(outdir, 'pedestrian_simulation.png'));

% KPI计算
disp('计算关键性能指标...');
kpi = calculate_kpi(selected, travelTime, rules);

fprintf('\n=== 最终KPI报告 ===\n');
fprintf('选中枢纽数量: %d\n', kpi.num_selected);
fprintf('30分钟覆盖率: %.1f%%\n', kpi.coverage_30min * 100);
fprintf('平均出行时间: %.1f 分钟\n', kpi.avg_travel_time);
fprintf('公平性指数: %.3f\n', kpi.fairness);
fprintf('总成本: %.0f\n', kpi.total_cost);

% 保存报告
save_kpi_report(kpi);
end

%% ===================== 重构的辅助函数 =====================

function T = fetch_osm_poi_refactored(center, radius_m)
% 重构的OSM数据获取
lat = center(1); lon = center(2);
q = sprintf(['[out:json][timeout:60];(node(around:%d,%f,%f)[amenity];', ...
             'node(around:%d,%f,%f)[shop];node(around:%d,%f,%f)[public_transport];', ...
             'node(around:%d,%f,%f)[tourism];node(around:%d,%f,%f)[highway~"bus_stop"];);out body;'], ...
             radius_m, lat, lon, radius_m, lat, lon, radius_m, lat, lon, radius_m, lat, lon);
url = 'http://overpass-api.de/api/interpreter';
options = weboptions('Timeout', 60, 'ContentType', 'text', 'CharacterEncoding', 'UTF-8');

try
    resp = webwrite(url, q, options);
    S = jsondecode(resp);
    
    elements = S.elements;
    if iscell(elements), elements = [elements{:}]; end
    
    isnode = arrayfun(@(x) isfield(x,'type') && strcmp(x.type,'node'), elements);
    nodes = elements(isnode);
    n = numel(nodes);
    
    latv = zeros(n,1); lonv = zeros(n,1); 
    name = cell(n,1); tags = cell(n,1);
    
    for i = 1:n
        latv(i) = nodes(i).lat;
        lonv(i) = nodes(i).lon;
        if isfield(nodes(i),'tags') && isfield(nodes(i).tags,'name')
            name{i} = nodes(i).tags.name;
        else
            name{i} = '';
        end
        if isfield(nodes(i),'tags')
            tags{i} = jsonencode(nodes(i).tags);
        else
            tags{i} = '';
        end
    end
    
    T = table(latv, lonv, name, tags, 'VariableNames', {'lat','lon','name','tags'});
    
catch ME
    error('OSM数据获取失败: %s', ME.message);
end
end

function dist = haversine_vectorized(lat1, lon1, lat2, lon2)
% 向量化Haversine距离计算
R = 6371;  % 地球半径(km)

phi1 = deg2rad(lat1);
phi2 = deg2rad(lat2);
delta_phi = deg2rad(lat2 - lat1);
delta_lambda = deg2rad(lon2 - lon1);

a = sin(delta_phi/2).^2 + cos(phi1).*cos(phi2).*sin(delta_lambda/2).^2;
c = 2 * atan2(sqrt(a), sqrt(1-a));
dist = R * c;
end

function F = evaluate_population_refactored(X, travelTime, rules, params, cand, cost_cap)
% 重构的种群评估函数 - 所有目标统一为最小化

N = size(X, 1);
F = zeros(N, 3);
valid_count = 0;

for i = 1:N
    x = X(i, :) > 0;
    if ~any(x), continue; end
    
    % 检查可行性
    [is_feasible, coverage] = check_solution_feasibility(x, travelTime, rules.min_coverage);
    if ~is_feasible, continue; end
    
    valid_count = valid_count + 1;
    
    % 目标1: 可达性 (越小越好)
    Tsel = travelTime(x, :);
    utility = -log(mean(exp(-params.BETA_LOGSUM * Tsel), 1)) / params.BETA_LOGSUM;
    f1 = -mean(utility);  % LogSumExp效用取负
    
    % 目标2: 成本 (越小越好)
    f2 = params.UNIT_COST * sum(x);
    
    % 检查成本约束
    if f2 > cost_cap, continue; end
    
    % 目标3: 公平性+分散度 (综合指标越小越好)
    acc_by_site = 1./(1 + mean(Tsel, 2));
    if params.USE_GINI
        fairness = 1 - gini_coefficient_refactored(acc_by_site);
    else
        fairness = 1 - std(acc_by_site);  % 标准差越小越公平
    end
    
    % 分散度奖励
    if sum(x) >= 2
        spacing = mean(pdist(cand(x, :)));
    else
        spacing = 0;
    end
    
    f3 = -(fairness + params.DISP_WEIGHT * spacing);  % 综合指标取负
    
    F(valid_count, :) = [f1, f2, f3];
end

F = F(1:valid_count, :);
end

function f = evaluate_single_solution_refactored(x, travelTime, rules, params, cand, kAllow, cost_cap)
% 评估单个解（用于GA）

xbin = round(x) > 0;
if sum(xbin) ~= kAllow
    f = [1e6, 1e9, 1e6];  % 惩罚不可行解
    return;
end

% 检查可行性
[is_feasible, ~] = check_solution_feasibility(xbin, travelTime, rules.min_coverage);
if ~is_feasible
    f = [1e6, 1e9, 1e6];
    return;
end

% 计算目标
Tsel = travelTime(xbin, :);
utility = -log(mean(exp(-params.BETA_LOGSUM * Tsel), 1)) / params.BETA_LOGSUM;
f1 = -mean(utility);

f2 = params.UNIT_COST * sum(xbin);
if f2 > cost_cap
    f = [1e6, 1e9, 1e6];
    return;
end

acc_by_site = 1./(1 + mean(Tsel, 2));
if params.USE_GINI
    fairness = 1 - gini_coefficient_refactored(acc_by_site);
else
    fairness = 1 - std(acc_by_site);
end

if sum(xbin) >= 2
    spacing = mean(pdist(cand(xbin, :)));
else
    spacing = 0;
end

f3 = -(fairness + params.DISP_WEIGHT * spacing);

f = [f1, f2, f3];
end

function [is_feasible, coverage] = check_solution_feasibility(x, travelTime, min_coverage)
% 严格的可行性检查
selected = find(x > 0);
if isempty(selected)
    is_feasible = false;
    coverage = 0;
    return;
end

Tsel = travelTime(selected, :);
coverage = mean(any(Tsel < 30, 1));  % 30分钟覆盖率
is_feasible = (coverage >= min_coverage);
end

function keep = is_pareto_efficient_refactored(F)
% 重构的Pareto非支配排序
N = size(F, 1);
keep = true(N, 1);

for i = 1:N
    if keep(i)
        % 找到被当前解支配的解
        dominated = all(F <= F(i, :), 2) & any(F < F(i, :), 2);
        keep(dominated) = false;
        keep(i) = true;
    end
end
end

function [best_idx, metrics] = select_best_solution(F, weights)
% 基于理想点法选择最终解

% 所有目标都是最小化，直接归一化
f1_norm = normalize_column(F(:, 1));  % 可达性
f2_norm = normalize_column(F(:, 2));  % 成本  
f3_norm = normalize_column(F(:, 3));  % 公平性

% 理想点 (0,0,0) - 所有目标都最优
dist_to_ideal = sqrt((weights(1)*f1_norm).^2 + ...
                     (weights(2)*f2_norm).^2 + ...
                     (weights(3)*f3_norm).^2);

[~, best_idx] = min(dist_to_ideal);

metrics.dist_to_ideal = dist_to_ideal(best_idx);
metrics.f1_norm = f1_norm(best_idx);
metrics.f2_norm = f2_norm(best_idx);
metrics.f3_norm = f3_norm(best_idx);
end

function z = normalize_column(x)
% 列归一化 [0,1]
z = (x - min(x)) / (max(x) - min(x) + eps);
end

function kpi = calculate_kpi(selected, travelTime, rules)
% 计算关键性能指标
Tsel = travelTime(selected, :);

kpi.num_selected = length(selected);
kpi.coverage_30min = mean(any(Tsel < 30, 1));
kpi.avg_travel_time = mean(min(Tsel, [], 1));
kpi.total_cost = rules.objective_weights(2) * kpi.num_selected;  % 简化成本计算

% 公平性
acc_by_site = 1./(1 + mean(Tsel, 2));
kpi.fairness = 1 - std(acc_by_site);

% 分散度
if kpi.num_selected >= 2
    kpi.mean_spacing = mean(pdist(Tsel'));
else
    kpi.mean_spacing = 0;
end
end

%% ===================== 其他辅助函数（保持原逻辑） =====================

function pts = generate_synthetic_POI(n)
centers = [0.2 0.3; 0.5 0.5; 0.8 0.7; 0.3 0.8];
pts = [];
for i = 1:size(centers,1)
    pts = [pts; bsxfun(@plus, centers(i,:), 0.05*randn(round(n/4),2))];
end
pts = min(max(pts,0),1);
end

function [lat_s, lon_s] = sample_kde_points_refactored(lat, lon, N)
lat = lat(:); lon = lon(:);
m = min(numel(lat), numel(lon)); lat = lat(1:m); lon = lon(1:m);
if numel(lat) < 50
    idx = randi(numel(lat), N,1);
    lat_s = lat(idx) + 0.0008*randn(N,1);
    lon_s = lon(idx) + 0.0008*randn(N,1);
    return;
end
bw_lat = 1.06*std(lat)*numel(lat)^(-1/5);
bw_lon = 1.06*std(lon)*numel(lon)^(-1/5);
idx = randi(numel(lat), N,1);
lat_s = lat(idx) + bw_lat*randn(N,1);
lon_s = lon(idx) + bw_lon*randn(N,1);
end

function [lat_s, lon_s] = sample_multimodal_hotspots_refactored(cand, N, geo_bounds)
K = min(size(cand,1), 6);
if K==0
    lat_s = geo_bounds.minlat + (geo_bounds.maxlat-geo_bounds.minlat)*rand(N,1);
    lon_s = geo_bounds.minlon + (geo_bounds.maxlon-geo_bounds.minlon)*rand(N,1);
    return;
end
idx = randi(K, N,1);
mu = cand(1:K,:);
lat0 = geo_bounds.minlat + mu(idx,2)*(geo_bounds.maxlat-geo_bounds.minlat);
lon0 = geo_bounds.minlon + mu(idx,1)*(geo_bounds.maxlon-geo_bounds.minlon);
lat_s = lat0 + 0.004*randn(N,1);
lon_s = lon0 + 0.004*randn(N,1);
end

function X = enumerate_exact_k(nCand, k)
C = nchoosek(1:nCand, k);
X = false(size(C,1), nCand);
for i=1:size(C,1), X(i,C(i,:)) = true; end
X = double(X);
end

function [pos,traj] = pedestrian_abm_refactored(selected,cand,nAgents,steps)
if isempty(selected), pos = rand(nAgents,2); traj = pos; return; end
targets = cand(selected,:);
pos = rand(nAgents,2); vel = zeros(nAgents,2);
traj = zeros(nAgents,2,steps);
for t = 1:steps
    desired = zeros(nAgents,2);
    for i = 1:nAgents
        d = sqrt(sum((targets - pos(i,:)).^2,2));
        [~,id] = min(d); goal = targets(id,:);
        desired(i,:) = goal - pos(i,:);
    end
    v_des = desired ./ max(sqrt(sum(desired.^2,2)),1e-3);
    rep = zeros(nAgents,2);
    for i=1:nAgents
        diff = pos - pos(i,:); dist = sqrt(sum(diff.^2,2));
        mask = (dist>0 & dist<0.05);
        rep(i,:) = sum(bsxfun(@times, diff(mask,:), (0.05-dist(mask))./0.05),1);
    end
    vel = 0.6*vel + 0.4*(v_des - 0.5*rep);
    pos = pos + 0.05*vel; traj(:,:,t) = pos;
end
end

function g = gini_coefficient_refactored(x)
x = x(:); x = x - min(x); x = x + eps; x = sort(x);
n = numel(x);
g = (2*sum((1:n)'.*x)/(n*sum(x))) - (n+1)/n;
end

function try_osm_basemap()
ok = false;
try, geobasemap('osm'); ok = true; catch
    templates = { ...
        'https://tile.openstreetmap.org/${z}/${x}/${y}.png', ...
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png', ...
        'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png' };
    for i = 1:numel(templates)
        try
            addCustomBasemap('osm', templates{i}, ...
                'Attribution','© OpenStreetMap contributors', 'DisplayName','OpenStreetMap');
            geobasemap('osm'); ok = true; break;
        catch, end
    end
end
if ~ok, error('无法注册/启用 OSM basemap（可能是网络/代理问题）。'); end
end

function [rep_idx, rep_F] = select_representative_solution(F)
% 为每个(k,cap)组合选择一个代表解
if size(F,1) == 1
    rep_idx = 1; rep_F = F;
    return;
end
% 选择距离理想点最近的解
f1_norm = normalize_column(F(:,1));
f2_norm = normalize_column(F(:,2));
f3_norm = normalize_column(F(:,3));
dist = sqrt(f1_norm.^2 + f2_norm.^2 + f3_norm.^2);
[~, rep_idx] = min(dist);
rep_F = F(rep_idx, :);
end

function save_pareto_results(Xpareto, Fpareto, selected, cand, metrics)
% 保存Pareto结果
global outdir;  % 添加这一行

writematrix(Xpareto, fullfile(outdir,'pareto_solutions_X.csv'));
writematrix(Fpareto, fullfile(outdir,'pareto_solutions_F.csv'));

% 选中的枢纽信息
if ~isempty(selected)
    hub_info = table(selected', cand(selected,1), cand(selected,2), ...
        'VariableNames', {'hub_id', 'x_norm', 'y_norm'});
    writetable(hub_info, fullfile(outdir,'selected_hubs.csv'));
end

% 选择指标
writetable(struct2table(metrics), fullfile(outdir,'selection_metrics.csv'));
end

function save_kpi_report(kpi)
% 保存KPI报告
global outdir;  % 添加这一行

kpi_table = struct2table(kpi);
writetable(kpi_table, fullfile(outdir,'final_kpi_report.csv'));

% 文本报告
fid = fopen(fullfile(outdir,'kpi_summary.txt'),'w');
fprintf(fid, 'VLA交通枢纽仿真 - 最终KPI报告\n');
fprintf(fid, '================================\n\n');
fprintf(fid, '选中枢纽数量: %d\n', kpi.num_selected);
fprintf(fid, '30分钟覆盖率: %.1f%%\n', kpi.coverage_30min * 100);
fprintf(fid, '平均出行时间: %.1f 分钟\n', kpi.avg_travel_time);
fprintf(fid, '公平性指数: %.3f\n', kpi.fairness);
fprintf(fid, '总成本: %.0f\n', kpi.total_cost);
fprintf(fid, '平均站间距: %.3f\n', kpi.mean_spacing);
fclose(fid);
end

function create_fallback_visualization(poi_table, cand_lat, cand_lon, selected, geo_bounds)
% 创建回退可视化
global outdir;  % 添加这一行

f = figure('Name', '地理可视化 (回退)', 'Position', [200 100 900 700]);
ax = geoaxes; hold(ax, 'on'); 
try 
    geobasemap(ax, 'streets'); 
catch
    % 如果所有底图都失败，使用无底图
end

geoscatter(ax, poi_table.lat, poi_table.lon, 6, 'filled', ...
    'MarkerFaceAlpha', 0.25, 'DisplayName', 'POI');
geoscatter(ax, cand_lat, cand_lon, 60, 'r', 'filled', ...
    'DisplayName', '候选点');

if ~isempty(selected)
    geoscatter(ax, cand_lat(selected), cand_lon(selected), 120, 'k', 'filled', ...
        'Marker', 'p', 'DisplayName', '选中枢纽');
end

geolimits(ax, [geo_bounds.minlat, geo_bounds.maxlat], ...
    [geo_bounds.minlon, geo_bounds.maxlon]);
title(ax, '北京南站区域 - 交通枢纽选址 (回退可视化)');

legend_labels = {'POI', '候选点'};
if ~isempty(selected)
    legend_labels{end+1} = '选中枢纽';
end
legend(ax, legend_labels, 'Location', 'northeastoutside');

saveas(f, fullfile(outdir, 'fallback_visualization.png'));
end
