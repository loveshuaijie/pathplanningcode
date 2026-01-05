#Prime.py为基元模型类的定义，三种基元模型：球体、圆柱体、长方体，应用：1、画出基元模型的三维图像 2、用于fcl的模型创建
from abc import abstractmethod
import numpy as np
import fcl
import matplotlib.pyplot as plt

class PrimeModel:
    @abstractmethod
    def draw(self, ax):  # 绘制模型的三维图像
        pass

    @abstractmethod
    def createmodelforfcl(self):  # 创建fcl模型
        pass

class Cuboid(PrimeModel):
    def __init__(self, size,centerPoint=np.array([0,0,0]), orientation=np.eye(3)):  # centerPoint是中心点坐标,默认为原点，size是长宽高，orientation是姿态矩阵，默认为3*3单位矩阵
        self.centerPoint = centerPoint
        self.size = size
        self.l, self.w, self.h = size
        self.orientation = orientation
        self.modelforfcl = self.createmodelforfcl()  # 创建用于fcl的模型
        self.type = 'cuboid'
        self.equivalentRadius=(self.l*self.l+self.w*self.w+self.h*self.h)**0.5/4

    def draw(self, ax):
        """绘制长方体"""
        center = self.centerPoint.reshape(3, 1)
        l, w, h = self.size

        x = [-l/2, l/2]
        y = [-w/2, w/2]
        z = [-h/2, h/2]
        matrix = self.orientation.dot(np.array([x, y, z])) + center.repeat(2, 1)
        x, y, z = matrix

        # 画六个面
        X = np.linspace(x[0], x[1], 5)
        Y = np.linspace(y[0], y[1], 5)
        Z = np.linspace(z[0], z[1], 5)

        # 上下面
        faceX, faceY = np.meshgrid(X, Y)
        upfaceZ = faceX * 0 + z[1]
        downfaceZ = faceX * 0 + z[0]
        ax.plot_surface(faceX, faceY, upfaceZ, color='red', alpha=0.5)
        ax.plot_surface(faceX, faceY, downfaceZ, color='red', alpha=0.5)

        # 左右面
        faceY, faceZ = np.meshgrid(Y, Z)
        leftfaceX = faceY * 0 + x[0]
        rightfaceX = faceY * 0 + x[1]
        ax.plot_surface(leftfaceX, faceY, faceZ, color='red', alpha=0.5)
        ax.plot_surface(rightfaceX, faceY, faceZ, color='red', alpha=0.5)

        # 前后面
        faceX, faceZ = np.meshgrid(X, Z)
        frontfaceY = faceX * 0 + y[1]
        backfaceY = faceX * 0 + y[0]
        ax.plot_surface(faceX, frontfaceY, faceZ, color='red', alpha=0.5)
        ax.plot_surface(faceX, backfaceY, faceZ, color='red', alpha=0.5)

    def createmodelforfcl(self):
        box = fcl.Box(self.l, self.w, self.h)
        transform = fcl.Transform()
        transform.setTranslation(np.asarray(self.centerPoint))
        transform.setRotation(self.orientation)  # orientation是旋转矩阵
        box_shape = fcl.CollisionObject(box, transform)
        return box_shape

class Cylinder(PrimeModel):
    def __init__(self, height, radius,centerPoint=np.array([0,0,0]), orientation=np.eye(3)): #第一个参数是圆柱的高度，第二个参数是圆柱的半径，orientation是姿态矩阵，默认为3*3单位矩阵
        self.centerPoint = centerPoint
        self.orientation = orientation
        self.size = [height, radius]
        self.height = height
        self.radius = radius
        self.type = 'cylinder'
        self.modelforfcl = self.createmodelforfcl()
        self.equivalentRadius = (self.height*self.height+self.radius*self.radius)**0.5/4

    def draw(self, ax):
        """绘制圆柱体"""
        center = self.centerPoint
        height = self.height
        radius = self.radius

        # 生成圆柱的底面和顶面
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(center[2] - height/2, center[2] + height/2, 20)
        r = np.linspace(0, radius, 20)
        R, T = np.meshgrid(r, theta)
        X = R * np.cos(T) + center[0]
        Y = R * np.sin(T) + center[1]
        theta_grid, z_grid = np.meshgrid(theta, z)
        z_bottom = z_grid * 0 + center[2] - height/2
        z_top = z_grid * 0 + center[2] + height/2

        x_bottom = center[0] + radius * np.cos(theta_grid)
        y_bottom = center[1] + radius * np.sin(theta_grid)

        # 绘制侧面
        x_side = center[0] + radius * np.cos(theta_grid)
        y_side = center[1] + radius * np.sin(theta_grid)
        bottom = [X, Y, z_bottom]
        top = [X, Y, z_top]
        side = [x_side, y_side, z_grid]
        ax.plot_surface(bottom[0], bottom[1], bottom[2], color='green', alpha=0.5)
        ax.plot_surface(top[0], top[1], top[2], color='green', alpha=0.5)
        ax.plot_surface(side[0], side[1], side[2], color='green', alpha=0.5)

    def createmodelforfcl(self):
        cylinder = fcl.Cylinder(self.radius, self.height)
        transform = fcl.Transform()
        transform.setTranslation(np.asarray(self.centerPoint))
        transform.setRotation(self.orientation)
        cylinder_shape = fcl.CollisionObject(cylinder, transform)
        return cylinder_shape

class Sphere(PrimeModel):
    def __init__(self,radius, centerPoint =np.array([0,0,0])): #第一个参数是球体的半径，centerPoint是球体的中心点
        self.type = 'sphere'
        self.centerPoint = centerPoint
        self.radius = radius
        self.size = [radius]
        self.modelforfcl = self.createmodelforfcl()
        self.equivalentRadius = radius

    def draw(self, ax):
        """绘制球体"""
        center = self.centerPoint
        radius = self.radius

        # 参数化球体
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='cyan', alpha=0.5)

    def createmodelforfcl(self):
        sphere = fcl.Sphere(self.radius)
        transform = fcl.Transform()
        transform.setTranslation(np.asarray(self.centerPoint))
        sphere_shape = fcl.CollisionObject(sphere, transform)
        return sphere_shape