from Prime import Cuboid,Cylinder,Sphere
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    model1=Cuboid([1,1,1],np.array([1,1,1]))
    model2=Cylinder(1,1)
    model3=Sphere(1)
    modellist=[model1,model2,model3]
    ax=plt.subplot(111,projection='3d')
    for i in range(3):
        modellist[i].draw(ax)
        plt.show()