#!/usr/bin/env python
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading
from tf.transformations import quaternion_matrix, quaternion_from_matrix,euler_from_matrix

class Logger:
    def __init__(self):
        self.f = open("/home/ldd/msckf_real/src/msckf_vio/test_vicon_gt/vicon_wxyz.txt", 'w')
        self.vicon_pose = [str(0),str(0),str(0),str(0),str(0),str(0), str(0)]
        
        self.cur_time = 0
        rospy.Subscriber("/vicon/firefly_sbx/firefly_sbx", TransformStamped,self.vicon_Cb)
        # rospy.Subscriber("/vrpn_client_node/jiahao3/pose",PoseStamped,self.opti_Cb)
        self.T_vicon_imu = np.array([[ 0.33638, -0.01749,  0.94156,  0.06901],
        [-0.02078, -0.99972, -0.01114, -0.02781],
        [0.94150, -0.01582, -0.33665, -0.12395],
        [0.0,      0.0,      0.0,      1.0]])
        print(euler_from_matrix(self.T_vicon_imu[:3,:3]))
        matrix = R.from_matrix(self.T_vicon_imu[:3,:3])
        euler = matrix.as_euler("zyx",degrees=True)
        print(euler)
        x = np.identity(4)
        x[0,3] = -1
        print(np.linalg.inv(x))
        # self.T_vicon_imu = np.linalg.inv(self.T_vicon_imu)
        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()

    def thread_job(self):
        rospy.spin()
        
    def write_data(self):
        self.f.write(str(self.cur_time))
        self.f.write(',')
        self.f.write(','.join(self.vicon_pose))
        self.f.write('\r\n')
        
    def write_title(self):
        self.f.write("time, vicon_x, vicon_y, vicon_z, vicon_q_x, vicon_q_y, vicon_q_z, vicon_q_w")
        self.f.write('\r\n')

    def vicon_Cb(self,msg):
        self.cur_time = msg.header.stamp.to_nsec()
        p = np.array([[msg.transform.translation.x],[msg.transform.translation.y],[msg.transform.translation.z]])
        q = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
        q = q/np.linalg.norm(q)
        T_vicon_w = np.identity(4)
        T_vicon_w[:3,:3] = quaternion_matrix(q)[:3,:3]
        print(T_vicon_w[:3,:3])
        matrix = R.from_quat(q)
        T_vicon_w[:3,:3] = matrix.as_matrix()
        print("R:", T_vicon_w[:3,:3])
        T_vicon_w[:3, 3] = p.reshape(1, -1)
        T_imu_w = T_vicon_w * np.linalg.inv(self.T_vicon_imu)
        matrix = np.identity(4)
        matrix[:3,:3] = T_imu_w[:3,:3]
        q_gt = quaternion_from_matrix(matrix)
        print(q_gt)
        q = R.from_matrix(matrix[:3,:3])
        q_gt = q.as_quat()
        print("R:", q_gt)
        self.vicon_pose =[str(T_imu_w[0,3]), str(T_imu_w[1,3]), str(T_imu_w[2,3]), str(q_gt[0]), str(q_gt[1]), str(q_gt[2]), str(q_gt[3])]
        
             
def main():
    print("start record!")
    rospy.init_node('vicon_node', anonymous=True)
    logger = Logger()
    rate = rospy.Rate(200)
    logger.write_title()
    while not rospy.is_shutdown():
        logger.write_data()
        rate.sleep()
    logger.f.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass