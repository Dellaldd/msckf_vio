#!/usr/bin/env python
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import Odometry

import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading
# from tf.transformations import quaternion_matrix, quaternion_from_matrix,euler_from_matrix

class Logger:
    def __init__(self):
        self.fold = "/home/ldd/bias_esti_ws/src/bias_esti/result/msckf/euroc_backend/v203/"
        self.f_gt = open(self.fold + "stamped_groundtruth.txt", 'w')
        self.f_gt_vel = open(self.fold + "groundtruth_velocity.txt", 'w')
        self.f_esti = open(self.fold + "stamped_traj_estimate.txt", 'w')
        self.f_esti_vel = open(self.fold + "traj_estimate_velocity.txt", 'w')
        
        self.gt_pose = []
        self.gt_vel = []
        self.esti_pose = []
        self.esti_vel = []
        
        
        rospy.Subscriber("/firefly_sbx/vio/odom", Odometry, self.esti_Cb)
        rospy.Subscriber("/firefly_sbx/vio/gt_odom", Odometry, self.gt_Cb)

        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()

    def thread_job(self):
        rospy.spin()
        
    def write_data(self):
        for data in self.gt_pose:
            if data[1] != "0.0":
                self.f_gt.write(' '.join(data))
                self.f_gt.write('\r\n')
        
        for data in self.gt_vel:
            self.f_gt_vel.write(' '.join(data))
            self.f_gt_vel.write('\r\n')
            
        for data in self.esti_pose:
            self.f_esti.write(' '.join(data))
            self.f_esti.write('\r\n')
        
        for data in self.esti_vel:
            self.f_esti_vel.write(' '.join(data))
            self.f_esti_vel.write('\r\n')
        
        
        
    def write_title(self):
        self.f_gt.write("# timestamp tx ty tz qx qy qz qw")
        self.f_gt.write('\r\n')
        self.f_esti.write("# timestamp tx ty tz qx qy qz qw")
        self.f_esti.write('\r\n')
        
        self.f_gt_vel.write("# timestamp tx ty tz qx qy qz qw vx vy vz")
        self.f_gt_vel.write('\r\n')
        self.f_esti_vel.write("# timestamp tx ty tz qx qy qz qw vx vy vz")
        self.f_esti_vel.write('\r\n')
        
        
        
    def gt_Cb(self, msg):
        self.gt_pose.append([str(msg.header.stamp.to_sec()), str(msg.pose.pose.position.x), str(msg.pose.pose.position.y),
            str(msg.pose.pose.position.z), str(msg.pose.pose.orientation.x), 
            str(msg.pose.pose.orientation.y),str(msg.pose.pose.orientation.z), 
            str(msg.pose.pose.orientation.w)])
        
        self.gt_vel.append([str(msg.header.stamp.to_sec()), str(msg.pose.pose.position.x), str(msg.pose.pose.position.y),
            str(msg.pose.pose.position.z), str(msg.pose.pose.orientation.x), 
            str(msg.pose.pose.orientation.y),str(msg.pose.pose.orientation.z), 
            str(msg.pose.pose.orientation.w), str(msg.twist.twist.linear.x), str(msg.twist.twist.linear.y),
            str(msg.twist.twist.linear.z)])
        
    
    def esti_Cb(self, msg):
        self.esti_pose.append([str(msg.header.stamp.to_sec()), str(msg.pose.pose.position.x), str(msg.pose.pose.position.y),
            str(msg.pose.pose.position.z), str(msg.pose.pose.orientation.x), 
            str(msg.pose.pose.orientation.y),str(msg.pose.pose.orientation.z), 
            str(msg.pose.pose.orientation.w)])
        
        self.esti_vel.append([str(msg.header.stamp.to_sec()), str(msg.pose.pose.position.x), str(msg.pose.pose.position.y),
            str(msg.pose.pose.position.z), str(msg.pose.pose.orientation.x), 
            str(msg.pose.pose.orientation.y),str(msg.pose.pose.orientation.z), 
            str(msg.pose.pose.orientation.w), str(msg.twist.twist.linear.x), str(msg.twist.twist.linear.y),
            str(msg.twist.twist.linear.z)])
        
        
def main():
    print("start record!")
    rospy.init_node('record_node', anonymous=True)
    logger = Logger()
    rate = rospy.Rate(200)
    logger.write_title()
    while not rospy.is_shutdown():
        rate.sleep()
    logger.write_data()
    logger.f_gt.close()
    logger.f_esti.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass