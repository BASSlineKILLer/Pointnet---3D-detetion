from pickle import TRUE
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack
import open3d as o3d
from .interpolation import Interpolation
class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        self.initial_num_points = model_cfg.SA_CONFIG.INITIAL_NUM_POINTS
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            self.SA_modules[-1].original_ratio = self.model_cfg.SA_CONFIG.NPOINTS[k] / self.initial_num_points
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, bias=False)
        # 第二层：256维输入，256维输出，使用1x1卷积
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(256,256, bias=True)
        self.linear2 = nn.Linear(512, 256, bias=True)  # 显式启用偏置
        self.linear3 = nn.Linear(512, 256, bias=True)  # 显式启用偏置
        
        self.linear4 = nn.Linear(256,128,bias =True)
    def knn(self,x, k):
        """
        Input:
            x: pointcloud data, [B, N, C]
            k: number of knn
        Return:
            pairwise_distance: distance of points, [B, N, N]
            idx: index of points' knn, [B, N, k]
        """
        x = x.transpose(2, 1) #[B, N, C]->[B, C, N]
        inner = -2*torch.matmul(x.transpose(2, 1), x) #(B, N, N)
        xx = torch.sum(x**2, dim=1, keepdim=True) #(B, 1, N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1) #(B, N, N)
        # print("pairwise_distance:",pairwise_distance.shape)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
        return -1*pairwise_distance, idx


    def process_point_cloud_mix(self,point_cloud, step_size, normals):
        ratio = 2 ** ((torch.rand(1) - 0.5) * 2)  # 将随机数映射到 [0.5, 2] 范围内      
        if ratio <= 1:
            processed_cloud = self.downsample_point_cloud_score(point_cloud, ratio)
        else:
            # Upsample the point cloud
            distance, idx_k = self.knn(point_cloud, k=20)
            processed_cloud = self.upsample_point_cloud(point_cloud, ratio, step_size,distance, idx_k, normals)
        return processed_cloud,processed_cloud.shape[1]

    def upsample_point_cloud(self,point_cloud, ratio, step_size,distance, idx_k, normals):
        """ Upsample the point cloud by adding noise and randomly selecting points. """
        # Simulate upsampling with noise and random selection
        num_points = point_cloud.shape[1]
        I = Interpolation(step_size)
        pc = I.random_k_neighbors_shape_invariant_perturb(point_cloud,distance, idx_k, normals)
        # Select random indices from the noise array
        num_points_to_select = int((ratio - 1) * num_points)
        sampled_indices = torch.randperm(num_points)[:num_points_to_select]

        # Concatenate selected noisy points with the original point cloud
        upsampled_cloud = torch.cat([point_cloud, pc[:, sampled_indices]], dim=1)

        return upsampled_cloud#,pc[:, sampled_indices]

    ###24-5-21 
    ###改进的drop方法，global_score决定drop点的(0.0-1.0)local-global范围
    ###origin
    # def farthest_point_sample_self(self,xyz, features, ratio):
    #     """
    #     Input:
    #         xyz: pointcloud coordinates, [B, N, 3]
    #         features: pointcloud features, [B, N, D]
    #         npoint: number of samples
    #     Return:
    #         sampled_xyz: sampled pointcloud coordinates, [B, npoint, 3]
    #         sampled_features: sampled pointcloud features, [B, npoint, D]
    #     """
    #     device = xyz.device
    #     B, N, C = xyz.shape
    #     D = features.shape[-1]
    #     npoint=int(N*ratio)
    #     # Initialize
    #     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    #     distance = torch.ones(B, N).to(device) * 1e10
    #     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    #     batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
    #     # For each sampling point
    #     for i in range(npoint):
    #         centroids[:, i] = farthest  # Store selected points indices
    #         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Get the coordinate of the farthest point
    #         dist = torch.sum((xyz - centroid) ** 2, -1)  # Calculate distances to the centroid point
    #         mask = dist < distance  # Find points closer to the current centroid
    #         distance[mask] = dist[mask]  # Update the minimum distance
    #         farthest = torch.max(distance, -1)[1]  # Select the farthest point
        
    #     # Get the coordinates and features of the sampled points
    #     sampled_xyz = xyz[batch_indices, centroids, :]  # Get the coordinates of sampled points
    #     sampled_features = features[batch_indices, centroids, :]  # Get the features of sampled points
        
    #     return sampled_xyz, sampled_features

    def farthest_point_sample_self(self, xyz, features, ratio):
        """
        Input:
            xyz: pointcloud coordinates, [B, N, 3]
            features: pointcloud features, [B, N, D]
            ratio: sampling ratio (e.g., 0.25 means sample 25% of points)
        Return:
            sampled_xyz: sampled pointcloud coordinates, [B, npoint, 3]
            sampled_features: sampled pointcloud features, [B, npoint, D]
        """
        device = xyz.device
        B, N, C = xyz.shape
        D = features.shape[-1]
        npoint = int(N * ratio)  # Number of points to sample
        # Initialize
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10  # Large initial distance
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # Random initialization of farthest points
        batch_indices = torch.arange(B, dtype=torch.long).to(device)  # Batch indices for all samples
        
        # For each sampling point
        for i in range(npoint):
            centroids[:, i] = farthest  # Store selected points indices
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Get the coordinate of the farthest point
            dist = torch.sum((xyz - centroid) ** 2, -1)  # Calculate squared distances to the centroid point
            mask = dist < distance  # Find points closer to the current centroid
            distance[mask] = dist[mask]  # Update the minimum distance for closer points
            farthest = torch.max(distance, -1)[1]  # Select the farthest point as the next centroid
        
        # Expand batch_indices to match the shape of centroids
        batch_indices = batch_indices.view(B, 1).expand(-1, npoint)  # [B, npoint]
        
        # Get the coordinates and features of the sampled points
        sampled_xyz = xyz[batch_indices, centroids, :]  # Get the coordinates of sampled points
        sampled_features = features[batch_indices, centroids, :]  # Get the features of sampled points
        
        return sampled_xyz, sampled_features




    #def downsample_point_cloud_score(point_cloud,r):
    def downsample_point_cloud_score(self,point_cloud,r):#,global_score):
        B, N, _ = point_cloud.shape
        k = int(N * (1 - r))
        # global_score=1
        global_score = torch.rand(1)#origin
        center_indices = torch.randint(0, N, (B,))

        center_points = point_cloud[torch.arange(B), center_indices]
        distances = torch.norm(point_cloud - center_points[:, None, :], dim=2)
        sorted_indices = distances.argsort(dim=1)

        new_k = int((N - k) * global_score) + k
        top_k_indices = sorted_indices[:, torch.randperm(new_k)[:k]]

        mask = torch.ones(B, N, dtype=torch.bool)
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, k)
        mask[batch_indices, top_k_indices] = False
        filtered_point_cloud = point_cloud[mask].view(B, N - k, 3)
        return filtered_point_cloud    
    def index_points(self,points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        # print("points:",points)
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        # print("batch:",batch_indices)
        new_points = points[batch_indices, idx, :]
        # print("new_points:",new_points)
        return new_points

    def square_distance(self,src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def knn_point(self,nsample, xyz, new_xyz):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        sqrdists = self.square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
        return group_idx
    def sample_and_group(self,npoint, radius, nsample, xyz, points):
        """
        Input:
            npoint:
            radius:
            nsample:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, nsample, 3]
            new_points: sampled points data, [B, npoint, nsample, 3+D]
        """

        B, N, C = xyz.shape
        S = npoint 
        xyz = xyz.contiguous()
        ###PointSP   
        fps_idx,idx_k = self.weighted_random_point_sample(xyz, npoint, k=20)
        new_xyz = self.index_points(xyz, fps_idx) ##xyz:(B,N,3)
        new_points = self.index_points(points, fps_idx)
        idx = self.knn_point(nsample, xyz, new_xyz) ###选周围点
        #idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, idx) # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
        grouped_points = self.index_points(points, idx)
        grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
        new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
        return new_xyz, new_points

    def weighted_random_point_sample(self,xyz, npoint, k=20,replace=False):
        """
        Input:
            xyz: pointcloud data, [B, N, C]
            npoint: number of samples
            k: k of knn
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        # print("k::::",k)
        device = xyz.device
        B, N, C = xyz.shape	
        centroids = torch.zeros(B, npoint, dtype=torch.long)#.to(device)
        # 计算采样权重
        import time
    
        weights, idx = self.cal_weight(xyz, k)  # 假设这个函数已经定义且返回(B, N)的weights和idx
        
        centroids = torch.multinomial(weights, npoint, replacement=replace).squeeze()  
        return centroids,idx


    def cal_weight(self,xyz, k=16):
        """
        Input:
            xyz: pointcloud data, [B, N, C]
            k: k of knn or max sample number in local region of ball_query
        Return:
            weights: weights of pointcloud, [B, N]
            idx: index of points' knn, [B, N, k]
        """
        B, N, C = xyz.shape

        pairwise_distance, idx = knn(xyz, k)
        #索引出k近邻的距离    
        distance = torch.gather(pairwise_distance, dim=-1, index=idx)#(B, N, k)
        # 沿着维度k计算平均值
        mean_along_k = torch.mean(distance, dim=-1)
        threshold = torch.quantile(mean_along_k, q=0.5, dim=-1)
        weights = torch.zeros_like(distance, dtype=torch.float32)  # shape: [B, N, k]
        weights[distance <= threshold.view(B, 1, 1)] = 1. #(B,N,k)
        weights = torch.sum(weights, dim=-1,keepdim=False) #(B, N)
        weights /= torch.sum(weights, dim=-1).view(B, 1)#/1024 # 防止除以零, (B, N) 
        return weights,idx


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features
    # def get_normal_vector(self,points):
    #     """Calculate the normal vector.
    #     Args:
    #         points (torch.cuda.FloatTensor): the point cloud with N points, [N, 3].
    #     """
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    #     normal_vec = torch.FloatTensor(np.asarray(pcd.normals)).cuda().unsqueeze(0)
    #     return normal_vec




    def get_normal_vector(self, points):
        """为批处理点云计算法向量
        参数:
            points (torch.cuda.FloatTensor): 点云形状为 [B, N, 3]
        返回:
            normals (torch.cuda.FloatTensor): 法向量形状为 [B, N, 3]
        """
        B, N, _ = points.shape
        normals_list = []
        for b in range(B):
            # 单独处理每个批次元素
            pcd = o3d.geometry.PointCloud()
            # 将单个批次元素转换为numpy数组 [N, 3]
            pc_np = points[b].detach().cpu().numpy()
            pcd.points = o3d.utility.Vector3dVector(pc_np)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            # 转换回张量并添加批次维度
            normals = torch.FloatTensor(np.asarray(pcd.normals)).cuda()
            normals_list.append(normals)
        # 堆叠所有批次元素
        normal_vec = torch.stack(normals_list, dim=0)
        return normal_vec    
    def interpolate_features(self,features, old_xyz, new_xyz):
        """
        :param features: (B, C, N) 特征
        :param old_xyz: (B, N, 3) 原始点云
        :param new_xyz: (B, M, 3) 新点云
        :return: (B, C, M) 插值后的特征
        """
        dist, idx = pointnet2_utils.three_nn(new_xyz,old_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(features, idx, weight)
        return interpolated_feats    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        # # (B, N, C)
        # x = xyz.permute(0, 2, 1)
        # batch_size, C, N = x.size()
        # # print("N:",int(N/2))
        # # B, D, N
        # x = F.relu(self.bn1(self.conv1(x)))
        # # B, D, N
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = x.permute(0, 2, 1)
        #x 卷积后每个点特征        
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        normals= self.get_normal_vector(xyz)
        old_xyz=xyz.clone()
        xyz,new_num_points= self.process_point_cloud_mix(xyz, 0.05,normals)
        
        features = self.interpolate_features(features, old_xyz, xyz)
        #xyz,features = self.sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)   
        #new_npoints=512
        new_npoints = [int(new_num_points * self.SA_modules[k].original_ratio) for k in range(len(self.SA_modules))]
        for i, sa_module in enumerate(self.SA_modules):
            # 根据原始比例计算新npoint
            original_ratio =  sa_module.npoint / self.initial_num_points
            #sa_module.npoint = max(1, int(new_num_points * original_ratio))
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], npoint=new_npoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )  # (B, C, N)
# FP 阶段：从最后一层反推，使用对应的 xyz 和 features
        # print("Layer shapes:")
        # for idx, (xyz, feat) in enumerate(zip(l_xyz, l_features)):
        #     print(f"Layer {idx}: xyz={xyz.shape}, feat={feat.shape}")

        # print("\nFP modules:", len(self.FP_modules))
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1],     # ← 使用的是前面保存的对应层的 xyz
                l_xyz[i],         # ← 上一层的 xyz（用于插值）
                l_features[i - 1], # ← 对应层的原始特征
                l_features[i]      # ← 上一层的特征（插值后）
            )



        #l_features[0]=l_features[0].permute(0,2,1)
        num_bf =l_features[0].shape[2]
        #print(l_features[0].shape,"features")
        l_features[1]=l_features[1].permute(0,2,1)
        l_features[2]=l_features[2].permute(0,2,1)
        l_features[3]=l_features[3].permute(0,2,1)
        l_features[0]=l_features[0].permute(1,0,2)
        #print(l_features[0].shape,'l_features[0]')
        #print(l_features[1].shape,'l_features[1]')
        #print(l_features[2].shape,'l_features[2]')
        #print(l_features[3].shape,'l_features[3]')
        ##MLP 128到256
        #
        l_features[0]=l_features[0].unsqueeze(0)#

        l_features[0] = self.conv3(l_features[0])

        

        
        l_features[0] = self.bn1(l_features[0])  # 批归一化
        l_features[0] = self.relu(l_features[0])  # 激活函数
        l_features[0]=l_features[0].squeeze(0)
        #print(l_features[0].shape,"features")
        l_features[0] =l_features[0].permute(1,2,0)
        #l_features[0] :(B,N,256)
        #print(l_features[0].shape,"features")
        ##
        #new_point1,new_point_features1=self.farthest_point_sample_self(l_xyz[0],l_features[0],1)#(B,N/4,256)
        new_point1,new_point_features1=self.farthest_point_sample_self(l_xyz[0],l_features[0],1)#(B,N/4,256)
        #print(new_point1.shape,'new_point1')
        #print(new_point_features1.shape,'new_point_features1')
        l_features[1] = self.linear1(l_features[1])
        #print(l_features[1].shape,"l_1")
        p_feature2=new_point_features1+l_features[1]#(B,N/4,256)
        #print(l_features[1].shape)
        #print(p_feature2.shape)
        
        new_point2,new_point_features2=self.farthest_point_sample_self(new_point1,p_feature2,0.25)
        l_features[2] = self.linear2(l_features[2])
        p_feature3 =new_point_features2+l_features[2]  #(B,N/16,256)
        #print(p_feature3.shape)

        new_npoints3,new_point_features3=self.farthest_point_sample_self(new_point2,p_feature3,0.25)
        l_features[3] = self.linear3(l_features[3])
        p_feature4 =new_point_features3+l_features[3]
        #print(p_feature4.shape)

        point_features = torch.cat((p_feature4, p_feature3, p_feature2, l_features[0]), dim=1)
        _,N_num,D = point_features.shape
        #print(point_features.shape,'point_features_s')
        #point_features = point_features.permute(0, 2, 1)
        #print(num_bf)
        # 使用1x1卷积替代线性层，输入通道为特征维度D
        sa = nn.Conv1d(N_num, num_bf, kernel_size=1)
        sa.to(point_features.device)
        point_features = sa(point_features)
        # 恢复维度顺序为 [B, N, 256]
        #point_features = point_features.permute(0, 2, 1)


        #print(point_features.shape,'point_features')
        point_features=self.linear4(point_features)
        B, N = l_xyz[0].shape[0], l_xyz[0].shape[1]
        #print(l_xyz[0].shape,'l_xyz[0]')
        point_features = point_features.contiguous()
        
        new_batch_idx = torch.arange(B, device=xyz.device).unsqueeze(1).repeat(1, N).view(-1)
        
        




        # point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        # B, N = l_xyz[0].shape[0], l_xyz[0].shape[1]
        # new_batch_idx = torch.arange(B, device=xyz.device).unsqueeze(1).repeat(1, N).view(-1)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((new_batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict



class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)



        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict
