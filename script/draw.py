import numpy as np
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def main():
    
    # T_gt1_w = np.identity(4)
    # euler = [0, 0, 0]
    # r = R.from_euler('zyx',euler)
    # T_gt1_w[:3, :3] = r.as_matrix()
    # T_gt1_w[0,3] = 1
    # T_gt1_w[1,3] = 1
    # T_gt1_w[2,3] = 1
    # print(T_gt1_w)
    
    # T_gt2_w = np.identity(4)
    # T_gt2_w[0,3] = 3
    # T_gt2_w[1,3] = 3
    # T_gt2_w[2,3] = 3
    
    # T_gt2_gt1 = np.dot(np.linalg.inv(T_gt1_w), T_gt2_w)
    # print(T_gt2_gt1)
    
    
    file_name = "imu_1.txt"
    save_name = "record_2.png"
    read_path = "/home/ldd/msckf_real/src/msckf_vio/imu/" + file_name
    data = np.loadtxt(read_path, delimiter=',', skiprows=1)
    fig, ax = plt.subplots(2, 3)
    eulers = []
    for i in range(data.shape[0]):
        q = [data[i,3], data[i,4], data[i,5], data[i,6]]
        euler = R.from_quat(q).as_euler('zyx', degrees=True)
        eulers.append(euler)
    eulers = np.array(eulers)
    
    ax[0][0].plot(data[:,0], 'b-')
    ax[0][1].plot(data[:,1], 'b-')
    ax[0][2].plot(data[:,2], 'b-')
    ax[1][0].plot(eulers[:,0], 'b-')
    ax[1][1].plot(eulers[:,1], 'b-')
    ax[1][2].plot(eulers[:,2], 'b-')
    
    ax[0, 0].set_title("position x(m)")
    ax[0, 1].set_title("position y(m)")
    ax[0, 2].set_title("position z(m)")
    ax[1, 0].set_title("roll(deg)")
    ax[1, 1].set_title("pitch(deg)")
    ax[1, 2].set_title("yaw(deg)")

    fig.legend()
    fig.tight_layout()
    save_path = "/home/ldd/msckf_vio/src/msckf_vio/path/" + save_name
    plt.savefig(save_path, dpi=300)
    plt.show()
  
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
