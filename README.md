# 暑期实习项目-激光雷达3D点云目标检测
## CIDI短期实习项目（2025.7）
## 实验内容
- 基于OpenPCDet框架进行模型设计与训练
- 对模型进行参数调优
- 对模型进行测试
### 一、实验目的
研究不同激光雷达反射强度归一化方法对 3D 目标检测模型性能的影响，通过对比四种归一化策略优化 PointRCNN 模型在 KITTI 数据集上的检测精度。

---

### 二、实验内容
1. ​**数据预处理优化**​：在 KITTI 数据集加载模块中实现四种反射强度归一化方法  
2. ​**模型配置调整**​：优化 PointRCNN 的 ROI 池化参数和学习率调度策略  
3. ​**性能评估**​：对比不同预处理方法在 Car、Pedestrian、Cyclist 三类目标上的 3D 检测 AP 值  

---

### 三、修改后的代码

#### 1. 数据预处理模块 (kitti_dataset.py)
```python
# Copyright [2022] [OpenMMLab]
# Copyright [2025] [BASSlineKILLer]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def get_lidar_min_max(self, idx):
    lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # 归一化反射强度到0~1
    points[:, 3] = points[:, 3] / 255.0
    return points

def get_lidar_zscore(self, idx):
    lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # Z-score归一化后映射到0~1
    intensity = points[:, 3]
    mean, std = np.mean(intensity), np.std(intensity)
    if std > 0:
        z_score = (intensity - mean) / std
        scaled = 0.5 + z_score / 6  # 假设99.7%的数据在±3σ内
        points[:, 3] = np.clip(scaled, 0.0, 1.0)
    else:
        points[:, 3] = 0.5
    return points

def get_lidar_log(self, idx):
    lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # 对数归一化
    intensity = points[:, 3]
    log_intensity = np.log(intensity + 1e-6)
    min_log, max_log = np.min(log_intensity), np.max(log_intensity)
    if max_log > min_log:
        points[:, 3] = (log_intensity - min_log) / (max_log - min_log)
    else:
        points[:, 3] = 0.0
    return points

def get_lidar_percent(self, idx):
    lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # 基于百分位数的归一化
    intensity = points[:, 3]
    p1, p99 = np.percentile(intensity, [1, 99])
    clipped = np.clip(intensity, p1, p99)
    points[:, 3] = (clipped - p1) / (p99 - p1)
    return points
```
#### 2. 模型配置文件 (pointrcnn.yaml)
```yaml
ROI_HEAD:
  NAME: PointRCNNHead
  CLASS_AGNOSTIC: True
  ROI_POINT_POOL:
    POOL_EXTRA_WIDTH: [0.0, 0.0, 0.0]
    NUM_SAMPLED_POINTS: 512  # 从128调整为512
  DEPTH_NORMALIZER: 70.0

OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 2
  NUM_EPOCHS: 80
  OPTIMIZER: adam_onecycle
  LR: 0.005  # 从0.01调整为0.005
  WEIGHT_DECAY: 0.01
  MOMENTUM: 0.9
```
#### 3. ROI 头实现 (pointrcnn_head.py)
```python
self.merge_down_layer = nn.Sequential(
    nn.Conv2d(c_out*2, c_out, kernel_size=1, bias=not use_bn),
    *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()],
    nn.Dropout(p=0.3)  # 新增dropout层防止过拟合
)
```
## 项目设计
### 创新点1：数据增强  
​**方法**​：  
![image1.png](pics\图片1.png)
通过动态混合采样策略实现上下采样的自适应控制，结合几何特征保持的扰动算法，平衡点云的局部细节和全局结构。  
- ​**动态采样比例**​：随机生成比例（0.5~2.0），调整点云密度  
- ​**下采样**​：基于距离加权的随机丢弃策略，避免特征丢失  
- ​**上采样**​：法向量引导的形状不变扰动，保持几何结构  
​**优势**​：  
计算复杂度优化（O(NlogN)），增强点云分类任务的鲁棒性和多样性。  

### 参数表  
| 参数名        | 作用                  | 取值范围      |
|---------------|-----------------------|-------------|
| ratio         | 控制采样比例          | [0.5, 2.0]  |
| step_size     | 扰动步长              | 0.03        |
| k (KNN)       | 邻域点数              | 20          |
| global_score  | 下采样全局权重        | [0.0, 1.0]  |

---

### 创新点2：点特征加权随机采样 
![image1.png](pics\图片2.png) 
​**核心假设**​：  
同层卷积中，与周围点特征差异大的点更可能是干扰点。  
​**方法**​：  
定义特征加权隔离率：  
$$ 
y_i = \max_{q_j \in \mathcal{N}_i^k} \left[ (r_{ij} + 1) \times \| p_i - q_j \|_2 \right] 
$$  
$$
w_i = \Pr_{d \in D_i}(d \geq \bar{y}), \quad 
\bar{y} = \operatorname{Median}\left( \{ y_i \}_{i=1}^N \right)
$$  
​**效果**​：  
降低干扰点的采样概率，提升下采样鲁棒性。  

---

### 创新点3：特征反向传播层（FP-backprop层）  
​**改进点**​：
![image1.png](pics\图片3.png)
在PointNet++的FP机制上构建**双向特征增强路径**，解决单向信息流导致的细节丢失问题。  
- ​**新增低→高层路径**​：通过下采样与横向连接传递细节（如P1→P4）  
- ​**特征构建块**​：下采样特征相加实现层级双向交互  
- ​**多尺度整合**​：保持通道一致并通过池化融合特征  
​**优势**​：  
增强细节感知能力，显著提升小目标检测表现。  

### 参数说明  
- ​**结构参数**​：  
  5个全连接层 + SA层（sample and grouping参数）  
- ​**特征图**​：  
  {P1-P4} 对应不同层级特征  

---

## 模型参数量实验  
```python
# Copyright [2022] [OpenMMLab]
# Copyright [2025] [BASSlineKILLer]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import torch

# 加载配置
cfg_from_yaml_file('cfgs/kitti_models/pointrcnn_iou.yaml', cfg)
logger = common_utils.create_logger()

# 构建模型
model = build_network(
    model_cfg=cfg.MODEL, 
    num_class=len(cfg.CLASS_NAMES), 
    dataset=dataset
)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数数量: {total_params:,}")  # 输出: 4,038,819
print(f"可训练参数数量: {trainable_params:,}")
