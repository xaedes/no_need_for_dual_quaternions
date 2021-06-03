import numpy as np
import quaternion

class tf:
    def __init__(self, r=quaternion.quaternion(1,0,0,0), t=(0,0,0)):
        self.rot = r.normalized()
        self.pos = np.array([*t])
        
    def __repr__(self):
        return str((self.rot, self.pos))
    
    def from_mat(mat4):
        return tf(quaternion.from_rotation_matrix(mat4[:3,:3]),mat4[:3,3])
    
    def mat(self):
        res = np.zeros((4,4),dtype=float)
        res[:3,:3] = quaternion.as_rotation_matrix(self.rot)
        res[:3,3] = self.pos
        res[3,3] = 1
        return res
    
    def mul_vec(self, vec):
        return self.pos + quaternion.as_rotation_matrix(self.rot) @ vec
    
    def mul_tf(self, other):
        result = tf()
        
        result.rot = quaternion.from_rotation_matrix( quaternion.as_rotation_matrix(self.rot) @ quaternion.as_rotation_matrix(other.rot) )
        result.pos = self.mul_vec(other.pos)
        
        return result
    
    def inverse(self):
        result = tf()
        result.rot = self.rot.inverse()
        result.pos = quaternion.as_rotation_matrix(result.rot) @ (-self.pos)
        return result
      
tf1 = tf(quaternion.from_euler_angles(1,2,3), (4,5,6))

tf2 = tf(quaternion.from_euler_angles(7,8,9), (10,11,12))

print(tf1)
print(tf.from_mat(tf1.mat()))

#(quaternion(-0.224845095366153, 0.708073418273571, 0.454648713412841, 0.491295496433882), array([4, 5, 6]))
#(quaternion(0.224845095366153, -0.708073418273571, -0.454648713412841, -0.491295496433882), array([4., 5., 6.]))

tf3 = tf1.mul_tf(tf2)

print(tf3)
print(tf.from_mat(tf1.mat() @ tf2.mat()))
#(quaternion(0.933158168131312, 0.117404989052047, 0.280211052203802, 0.192129300259852), array([20.44659274, 13.07069146, 11.41973492]))
#(quaternion(0.933158168131312, 0.117404989052047, 0.280211052203802, 0.192129300259852), array([20.44659274, 13.07069146, 11.41973492]))

print(tf3.mul_vec((3,4,5)))
print((tf3.mat() @ (3,4,5,1))[:3])
#array([24.42327223, 17.38101516, 15.37032093])
#array([24.42327223, 17.38101516, 15.37032093])

print(tf3.inverse())
print(tf.from_mat(np.linalg.inv(tf3.mat())))
print(tf3.inverse().mat())
print(np.linalg.inv(tf3.mat()))
#(quaternion(0.933158168131312, -0.117404989052047, -0.280211052203802, -0.192129300259852), array([-15.81612497,  -9.49091609, -19.47021259]))
#(quaternion(-0.933158168131312, 0.117404989052047, 0.280211052203802, 0.192129300259852), array([-15.81612497,  -9.49091609, -19.47021259]))
#array([[  0.7691362 ,   0.4243704 ,  -0.47784859, -15.81612497],
#       [ -0.2927777 ,   0.8986048 ,   0.32678836,  -9.49091609],
#       [  0.56807634,  -0.11144134,   0.81539567, -19.47021259],
#       [  0.        ,   0.        ,   0.        ,   1.        ]])
#array([[  0.7691362 ,   0.4243704 ,  -0.47784859, -15.81612497],
#       [ -0.2927777 ,   0.8986048 ,   0.32678836,  -9.49091609],
#       [  0.56807634,  -0.11144134,   0.81539567, -19.47021259],
#       [  0.        ,   0.        ,   0.        ,   1.        ]])
#quaternion.quaternion(1,0,0,0)
