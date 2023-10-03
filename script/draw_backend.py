import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion
import math

def quaternion_to_euler(q, degree_mode=1):
    qw, qx, qy, qz = q

    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = math.asin(2 * (qw * qy - qz * qx))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    # degree_mode=1:【输出】是角度制，否则弧度制
    if degree_mode == 1:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
    euler = np.array([roll, pitch, yaw])
    return euler


def main():
    
    foldpath = "/home/ldd/bias_esti_ws/src/bias_esti/result/msckf/simulation/bias_noise/"
    gt_path = foldpath + "groundtruth_velocity.txt"
    esti_path = foldpath + "traj_estimate_velocity.txt"
    bias_path = foldpath + "bias.txt"
    
    save_position_path = foldpath + "position.png"
    save_bias_path = foldpath + "bias.png"
    
    gt = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    esti = np.loadtxt(esti_path, delimiter=' ', skiprows=1)
    bias = np.loadtxt(bias_path, delimiter=' ', skiprows=1)
    
    print(gt.shape)
    eulers = []
    eulers_gt = []
    error_pos = []
    
    end = min(esti.shape[0], gt.shape[0])
    start = 0
    for i in range(gt.shape[0]):
        q = [esti[i,4], esti[i,5], esti[i,6], esti[i,7]]
        # euler = R.from_quat(q).as_euler("xyz", degrees=True)
        euler = list(euler_from_quaternion(q))
        if euler[0] > 0:
                euler[0] -= np.pi
        else:
            euler[0] += np.pi
                
        # euler = quaternion_to_euler(q)
        eulers.append(euler)
        
        q_gt = [gt[i,4], gt[i,5], gt[i,6], gt[i,7]]
        # euler_gt = R.from_quat(q_gt).as_euler("xyz", degrees=True)
        euler_gt = list(euler_from_quaternion(q_gt))
        if euler_gt[0] > 0:
                euler_gt[0] -= np.pi
        else:
            euler_gt[0] += np.pi
        eulers_gt.append(euler_gt)
        
        error_pos.append(np.array([gt[i,1]-esti[i,1], gt[i,2]-esti[i,2], gt[i,3]-esti[i,3]]))
        
    error_pos = np.array(error_pos)
    error_pos = np.multiply(error_pos, error_pos)
    error_pos = np.sum(error_pos, axis=1)
    print(np.sum(np.sqrt(error_pos))/error_pos.shape[0]) 
    
    eulers_gt = np.array(eulers_gt)
    eulers = np.array(eulers)
    fig1, ax1 = plt.subplots(3, 3)
    
    ax1[0][0].plot(esti[:end,0], esti[:end,1], 'b-', label = 'esti')
    ax1[0][1].plot(esti[:end,0], esti[:end,2], 'b-', label = 'esti')
    ax1[0][2].plot(esti[:end,0], esti[:end,3], 'b-', label = 'esti')
    
    ax1[1][0].plot(esti[:end,0], eulers[:end,0], 'b-', label = 'esti')
    ax1[1][1].plot(esti[:end,0], eulers[:end,1], 'b-', label = 'esti')
    ax1[1][2].plot(esti[:end,0], eulers[:end,2], 'b-', label = 'esti')
    
    ax1[2][0].plot(esti[:end,0], esti[:end,8], 'b-', label = 'esti')
    ax1[2][1].plot(esti[:end,0], esti[:end,9], 'b-', label = 'esti')
    ax1[2][2].plot(esti[:end,0], esti[:end,10], 'b-', label = 'esti')
    
    ax1[0][0].plot(gt[:end,0], gt[:end,1], 'r-', label = 'gt')
    ax1[0][1].plot(gt[:end,0], gt[:end,2], 'r-', label = 'gt')
    ax1[0][2].plot(gt[:end,0], gt[:end,3], 'r-', label = 'gt')
    
    ax1[1][0].plot(gt[:end,0], eulers_gt[:end,0], 'r-', label = 'gt')
    ax1[1][1].plot(gt[:end,0], eulers_gt[:end,1], 'r-', label = 'gt')
    ax1[1][2].plot(gt[:end,0], eulers_gt[:end,2], 'r-', label = 'gt')
    
    ax1[2][0].plot(gt[:end,0], gt[:end,8], 'r-', label = 'gt')
    ax1[2][1].plot(gt[:end,0], gt[:end,9], 'r-', label = 'gt')
    ax1[2][2].plot(gt[:end,0], gt[:end,10], 'r-', label = 'gt')
    
    ax1[0, 0].set_title("position x(m)")
    ax1[0, 1].set_title("position y(m)")
    ax1[0, 2].set_title("position z(m)")
    
    ax1[1, 0].set_title("roll(rad)")
    ax1[1, 1].set_title("pitch(rad)")
    ax1[1, 2].set_title("yaw(rad)")
    
    ax1[2, 0].set_title("velocity x(m/s)")
    ax1[2, 1].set_title("velocity y(m/s)")
    ax1[2, 2].set_title("velocity z(m/s)")
    
    lines, labels = fig1.axes[-1].get_legend_handles_labels()
    fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
    # plt.legend()
    fig1.tight_layout()
    plt.savefig(save_position_path, dpi=300)

    fig2, ax2 = plt.subplots(4, 3)
    ax2[0][0].plot(bias[start:end,0], bias[start:end,1], 'r-')
    ax2[0][1].plot(bias[start:end,0], bias[start:end,2], 'r-')
    ax2[0][2].plot(bias[start:end,0], bias[start:end,3], 'r-')
    
    ax2[1][0].plot(bias[start:end,0], bias[start:end,4], 'r-')
    ax2[1][1].plot(bias[start:end,0], bias[start:end,5], 'r-')
    ax2[1][2].plot(bias[start:end,0], bias[start:end,6], 'r-')
    
    ax2[2][0].plot(bias[start:end,0], bias[start:end,7], 'r-')
    ax2[2][1].plot(bias[start:end,0], bias[start:end,8], 'r-')
    ax2[2][2].plot(bias[start:end,0], bias[start:end,9], 'r-')
    
    ax2[3][0].plot(bias[start:end,0], bias[start:end,10], 'r-')
    ax2[3][1].plot(bias[start:end,0], bias[start:end,11], 'r-')
    
    
    
    
    ax2[0, 0].set_title("acc_x(m^s-2)")
    ax2[0, 1].set_title("acc_y(m^s-2)")
    ax2[0, 2].set_title("acc_z(m^s-2)")
    
    ax2[1, 0].set_title("gyro_x(rad/s)")
    ax2[1, 1].set_title("gyro_y(rad/s)")
    ax2[1, 2].set_title("gyro_z(rad/s)")
    
    ax2[2, 0].set_title("imu used to integrate")
    ax2[2, 1].set_title("imu buffer size")
    ax2[2, 2].set_title("features used to optimize")
    
    ax2[3, 0].set_title("bias_a(m^s-2)")
    ax2[3, 1].set_title("bias_w(rad/s)")
    
    plt.legend()
    fig2.tight_layout()
    plt.savefig(save_bias_path, dpi=300)
    
    plt.show()
  
    

if __name__ == '__main__':

    main()
    
