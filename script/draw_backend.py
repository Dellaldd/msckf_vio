import numpy as np
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def main():
    fold = "/home/ldd/msckf_real/src/msckf_vio/backend_result/simulation/simulation_bias/"
    gt_file_name = "stamped_groundtruth.txt"
    esti_file_name = "stamped_traj_estimate.txt"
    save_name = "backend.png"
    esti_path = fold + esti_file_name
    gt_path = fold + gt_file_name
    
    save_path = fold  + save_name
    
    data_gt = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    data_esti = np.loadtxt(esti_path, delimiter=' ', skiprows=1)
    fig, ax = plt.subplots(2, 3)
    

    end = min(data_gt.shape[0], data_esti.shape[0])
    
    
    # data[:,0] = data[:,0] / 1e9
    # data[:,8] = data[:,8] / 1e9
    
    esti = []
    gt = []
    for i in range(end):
        q = [data_esti[i, 4],data_esti[i, 5], data_esti[i, 6], data_esti[i, 7]]
        euler = R.from_quat(q).as_euler("xyz", degrees= True)
        esti.append(np.array([data_esti[i,0], data_esti[i,1], data_esti[i,2], data_esti[i,3], euler[0], euler[1], euler[2]]))
        
        q = [data_gt[i, 4], data_gt[i, 5], data_gt[i, 6], data_gt[i, 7]]
        euler = R.from_quat(q).as_euler("xyz", degrees= True)
        
        gt.append(np.array([data_gt[i,0], data_gt[i,1], data_gt[i,2], data_gt[i,3], euler[0], euler[1], euler[2]]))
        
    esti = np.array(esti)
    gt = np.array(gt)
    
    ax[0][0].plot(esti[:, 0], esti[:,1], 'b-', label = "esti")
    ax[0][1].plot(esti[:, 0], esti[:,2], 'b-')
    ax[0][2].plot(esti[:, 0], esti[:,3], 'b-')
    
    ax[1][0].plot(esti[:, 0], esti[:,4], 'b-')
    ax[1][1].plot(esti[:, 0], esti[:,5], 'b-')
    ax[1][2].plot(esti[:, 0], esti[:,6], 'b-')

    ax[0][0].plot(gt[:, 0], gt[:,1], 'r-', label = "opti")
    ax[0][1].plot(gt[:, 0], gt[:,2], 'r-')
    ax[0][2].plot(gt[:, 0], gt[:,3], 'r-')
    
    ax[1][0].plot(gt[:, 0], gt[:,4], 'r-')
    ax[1][1].plot(gt[:, 0], gt[:,5], 'r-')
    ax[1][2].plot(gt[:, 0], gt[:,6], 'r-')
    
    ax[0, 0].set_title("position x(m)")
    ax[0, 1].set_title("position y(m)")
    ax[0, 2].set_title("position z(m)")
    ax[1, 0].set_title("roll(deg)")
    ax[1, 1].set_title("pitch(deg)")
    ax[1, 2].set_title("yaw(deg)")

    fig.legend()
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.show()
  
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
