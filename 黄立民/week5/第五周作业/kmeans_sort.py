import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

        # print("point",self.points)

    def cluster(self):
        # print(self.points)
        result = []
        distances = []
        for i in range(self.cluster_num):
            result.append([])
            distances.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i

            index1 = -1   #初始状态  距离最近的中心点没有聚集点
            for j in range(len(result[index])):

                if(distance_min <  distances[index][j]):   #找到插入位置

                    distances[index] = distances[index][0:j] + [distance_min.tolist()] + distances[index][j:]   #插入排序
                    result[index] = result[index][0:j] + [item.tolist()] + result[index][j:]                 #插入排序

                    index1 = 1
                    break

            if index1 == -1:       # 距离最近的中心点没有聚集点  或距离本来就是“类中“距离最大的点
                result[index] = result[index] + [item.tolist()]
                distances[index] = distances[index] + [distance_min.tolist()]
            #
            # result[index] = result[index] + [item.tolist()]
            # distances[index] = distances[index] + [distance_min.tolist()]



        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())

        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum, distances
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        # print("index",indexes)
        points = []
        for index in indexes:
            # print(index)
            # print(ndarray[index])
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(10, 3)
print(x.shape)
kmeans = KMeansClusterer(x, 3)
result, centers, distances,min_distances = kmeans.cluster()
print(result)
print(centers)
print(distances)

print(min_distances)
