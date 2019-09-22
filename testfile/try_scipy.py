from scipy.spatial.transform import Rotation as R
import numpy as np

t_matrix=np.zeros((4,4))
translation_vector=np.array([1,2,3])

quaterion_ex=[ -0.545621, 0.6779879, 0.2008292, 0.4497752 ]
r = R.from_quat(quaterion_ex)
r_matrix=r.as_dcm()
t_matrix[0:3,0:3]=r_matrix
t_matrix[0:3,3]=translation_vector.T
t_matrix[3,:]=np.array([0,0,0,1])
print(t_matrix)
print(r_matrix)
