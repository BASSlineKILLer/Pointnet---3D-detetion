import mayavi.mlab as mlab
import numpy as np
import torch

# 关键：开启Mayavi离线渲染模式，适配无GUI的终端环境
mlab.options.offscreen = True

box_colormap = [
    [1, 1, 1],    # 白色
    [0, 1, 0],    # 绿色
    [0, 1, 1],    # 青色
    [1, 1, 0]     # 黄色
]


def check_numpy_to_torch(x):
    """将numpy数组转换为torch张量（如需）"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """沿Z轴旋转点云"""
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    # 旋转矩阵
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    # 旋转点云（仅x,y,z坐标）
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """将3D边界框转换为8个顶点坐标"""
    # 边界框格式：(x,y,z,dx,dy,dz,heading)，中心坐标+长宽高+旋转角
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    # 8个顶点的模板（相对于中心的偏移）
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2  # 除以2是因为dx是全长，偏移为半长

    # 计算顶点（先缩放再旋转再平移）
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]  # 缩放
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)  # 旋转
    corners3d += boxes3d[:, None, 0:3]  # 平移到中心坐标

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(1024, 768), draw_origin=True):
    """可视化点云"""
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    # 绘制点云（调整点大小为0.1，避免重叠）
    if show_intensity and pts.shape[1] >=4:
        # 按强度着色
        mlab.points3d(
            pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], 
            mode='point', colormap='gnuplot', 
            scale_factor=0.1, figure=fig
        )
    else:
        # 白色点云
        mlab.points3d(
            pts[:, 0], pts[:, 1], pts[:, 2], 
            mode='point', color=(1, 1, 1),
            scale_factor=0.1, figure=fig
        )

    # 绘制原点坐标系（x:红, y:绿, z:蓝）
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)  # 原点
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=0.05, figure=fig)  # x轴（红）
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.05, figure=fig)  # y轴（绿）
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(0, 0, 1), tube_radius=0.05, figure=fig)  # z轴（蓝）

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=0.02, color=(0.5, 0.5, 0.5)):
    """绘制网格线"""
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    """绘制多网格范围"""
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)
    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, output_path="output.png"):
    """绘制场景（点云+框+网格）并保存图片"""
    # 转换为numpy数组（确保在CPU上）
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()
    print(111)
    # 初始化场景
    fig = visualize_pts(points, size=(1024, 768))  # 更大的画布
    fig = draw_multi_grid_range(fig, bv_range=(-60, -60, 60, 60))  # 调整网格范围
    print(111)
    # 绘制真实框（蓝色）
    if gt_boxes is not None and len(gt_boxes) > 0:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)  # 蓝色
    print(111)
    # 绘制预测框（按标签着色）
    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            # 无标签时默认绿色
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            # 按标签索引取模着色，避免越界
            unique_labels = np.unique(ref_labels)
            for label in unique_labels:
                label_idx = label % len(box_colormap)
                cur_color = tuple(box_colormap[label_idx])
                mask = (ref_labels == label)
                fig = draw_corners3d(
                    ref_corners3d[mask], 
                    fig=fig, 
                    color=cur_color, 
                    cls=ref_scores[mask] if ref_scores is not None else None, 
                    max_num=100
                )

    # 调整视角（更清晰的角度）
    mlab.view(azimuth=-45, elevation=45, distance=50, roll=0)

    # 保存图片并关闭，释放资源
    mlab.savefig(output_path)
    mlab.close(fig)  # 关闭当前画布
    print(f"场景图像已保存至：{output_path}")
    return output_path


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=0.05):
    """绘制3D边界框的8个顶点连接的线框"""
    num = min(max_num, len(corners3d))  # 限制最大绘制数量
    for n in range(num):
        b = corners3d[n]  # 单个框的8个顶点 (8,3)

        # 绘制分数标签（如果有）
        if cls is not None and len(cls) > n:
            score = cls[n]
            # 在框的顶部顶点显示分数
            mlab.text3d(
                b[4, 0], b[4, 1], b[4, 2],  # 选择顶部顶点（索引4）
                f"{score:.2f}", 
                color=color, 
                scale=0.3, 
                figure=fig
            )

        # 绘制框的12条边
        edges = [
            # 底面
            [0, 1], [1, 2], [2, 3], [3, 0],
            # 顶面
            [4, 5], [5, 6], [6, 7], [7, 4],
            # 连接底面和顶面
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        for (i, j) in edges:
            mlab.plot3d(
                [b[i, 0], b[j, 0]], 
                [b[i, 1], b[j, 1]], 
                [b[i, 2], b[j, 2]], 
                color=color, 
                tube_radius=tube_radius,
                line_width=line_width, 
                figure=fig
            )
    return fig