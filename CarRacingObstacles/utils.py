import numpy as np


def project_polygon(axis, polygon):
    """ 投影多邊形到軸上，返回最小和最大的投影值 """
    dots = np.dot(polygon, axis)
    return np.min(dots), np.max(dots)


def polygons_intersect(polygon1, polygon2):
    polygon1 = np.array(polygon1)
    polygon2 = np.array(polygon2)
    """ 使用分離軸定理檢測兩個多邊形是否碰撞 """
    for polygon in [polygon1, polygon2]:
        num_points = len(polygon)
        for i in range(num_points):
            # 取得邊
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_points]
            # 計算法向量
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])  # Perpendicular vector
            axis = axis / np.linalg.norm(axis)  # 正規化

            # 投影多邊形到法向量上
            min1, max1 = project_polygon(axis, polygon1)
            min2, max2 = project_polygon(axis, polygon2)

            # 檢查投影是否重疊
            if max1 < min2 or max2 < min1:
                return False
    return True