import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_matrix, euler_from_quaternion, quaternion_from_euler
from mpl_toolkits.mplot3d import Axes3D
import random
class CamFeature:
    def __init__(self):
        self.time = 0
        self.observation = {}
        
class DroneControlSim:
    def __init__(self):
        self.noise_w = 0
        self.feature_noise = 0
        self.sim_time = 100
        self.sim_step = 0.0025
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.accel = []
        self.imu_dataset = []
        self.cam_features = []
        self.feature3d = []
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.g_rot = -9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])
        
        self.intrinsics0 = [458.654, 457.296, 367.215, 248.375]
        self.K0 = np.array([[self.intrinsics0[0], 0.0, self.intrinsics0[2], 0],
            [0.0, self.intrinsics0[1], self.intrinsics0[3], 0],
            [0.0, 0.0, 1.0, 0]])
        
        # self.T_cam0_imu = np.identity(4)
        self.T_cam0_imu = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
            [0.0, 0.0, 0.0, 1.0]])
    
        
        self.intrinsics1 = [457.587, 456.134, 379.999, 255.238]
        self.K1 = np.array([[self.intrinsics1[0], 0.0, self.intrinsics1[2], 0],
            [0.0, self.intrinsics1[1], self.intrinsics1[3], 0],
            [0.0, 0.0, 1.0, 0]])
        # self.T_cam1_imu = np.identity(4)
        self.T_cam1_imu = np.array([[0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
            [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
            [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
            [0.0, 0.0, 0.0, 1.0]])

    def euler_2_quat(self, phi, theta, psi):
        rotation = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])
        quat = R.from_matrix(rotation.transpose()).as_quat()
        quat /= np.linalg.norm(quat)
        return quat
    
    def drone_dynamics(self,T,M):
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])/self.m# R_E_B: from earth to body
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position, d_velocity, d_angle,d_q))

        return dx 

    def run(self):
        position_target = []
        count = 0
        for i in np.arange(0, np.pi*25, np.pi/100):
            position_target.append(np.array([np.sin(i), 2*np.sin(i)+0.001*i, 1 + 0.5* np.sin(i)]))
            # position_target.append(np.array([1, 1, 1.5]))
            count += 1
        position_id = 0
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            if (self.pointer % 20 == 0 and self.pointer != 0):
                position_id += 1
            self.position_cmd[self.pointer] = position_target[position_id]

                
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])
            
            # self.velocity_cmd[self.pointer] = [0.0,0.0,-1.0]
            pitch_roll_cmd,thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            self.attitude_cmd[self.pointer] = np.append(pitch_roll_cmd,psi_cmd)

            #self.attitude_cmd[self.pointer] = [1,0,0]
            self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])

            # self.rate_cmd[self.pointer] = [1,0,0]
            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m

            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.016 
        kp_q = 0.016 
        kp_r = 0.028 
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

    def attitude_controller(self,cmd):
        kp_phi = 2.5 
        kp_theta = 2.5 
        kp_psi = 2.5
        error = cmd - self.drone_states[self.pointer,6:9]
        return np.array([kp_phi*error[0],kp_theta*error[1],kp_psi*error[2]])

    def velocity_controller(self,cmd):
        kp_vx = -0.2
        kp_vy = 0.2
        kp_vz = 2

        psi = self.drone_states[self.pointer,8]
        R = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        error = R@(cmd - self.drone_states[self.pointer,3:6])
        return np.array([kp_vy*error[1],kp_vx*error[0]]),kp_vz*error[2]-self.g*self.m

    def position_controller(self,cmd):
        kp_x = 0.7 
        kp_y = 0.7 
        kp_z = 0.7 

        error = cmd - self.drone_states[self.pointer,0:3]
        return np.array([kp_x*error[0],kp_y*error[1],kp_z*error[2]])


    def plot_states(self):
        self.accel = np.array(self.accel)
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        # ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.accel[:,0])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        
        # ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.accel[:,1])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        
        # ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.accel[:,2])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')
        
    def generate_cam_feature(self):
        Rot = np.identity(3)
        Rot[2,2] = -1
        Rot[1,1] = -1
        for i in np.arange(self.time.shape[0]):
            if i == 0:
                self.imu_dataset.append(np.array([self.time[i], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,self.g_rot]))
                self.accel.append(np.array([0,0,self.g_rot]))
            else:
                vel = Rot@self.drone_states[i, 3:6]
                velocity_0 = Rot@self.drone_states[i-1, 3:6]
                pos = Rot@self.drone_states[i, :3] 
                
                # q = self.euler_2_quat(self.drone_states[i, 6], self.drone_states[i, 7], self.drone_states[i, 8])
                q = quaternion_from_euler(self.drone_states[i, 6], -self.drone_states[i, 7], -self.drone_states[i, 8])
                R_imu_w = R.from_quat(q).as_matrix()
                quat = R.from_matrix(R_imu_w).as_quat()
                
                acc_w = (vel - velocity_0)/(self.time[i]-self.time[i-1])
                acc =  acc_w - np.array([0,0,self.g_rot])
                # acc = (vel - velocity_0)/(self.time[i]-self.time[i-1]) + np.array([0,0,self.g])
                
                acc = np.dot(R_imu_w.transpose(), acc)
                
                gyro = Rot@self.drone_states[i,9:]
                state = np.array([self.time[i], quat[3], quat[0], quat[1], quat[2], pos[0], pos[1], pos[2],  
                                    vel[0], vel[1], vel[2],
                                    gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2]])
                self.imu_dataset.append(state)
                self.accel.append(acc)
        
        
        x = np.arange(-1, 1.5, 0.5)
        y = np.arange(-1, 1.5, 0.5)
        z = np.arange(3, 5, 0.5)
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    self.feature3d.append(np.array([x[i], y[j], z[k], 1]))
                   
        print("feature num: ", len(self.feature3d))

        
        for i in np.arange(len(self.imu_dataset)):
            if (i % 20 == 0):
                cam_feature = CamFeature()
                cam_feature.time = self.imu_dataset[i][0]
                feature = {}
                pos = self.imu_dataset[i][5:8]
                quat = [self.imu_dataset[i][2], self.imu_dataset[i][3], self.imu_dataset[i][4], self.imu_dataset[i][1]]
                T_imu_w = np.identity(4)
                T_imu_w[:3, :3] = R.from_quat(quat).as_matrix()
                T_imu_w[:3, 3] = pos
                T_cam0_w = np.dot(T_imu_w, self.T_cam0_imu)
                T_cam1_w = np.dot(T_imu_w, self.T_cam1_imu)
                
                for id in range(len(self.feature3d)):
                    pc0 = np.dot(np.linalg.inv(T_cam0_w), self.feature3d[id])
                    pc1 = np.dot(np.linalg.inv(T_cam1_w), self.feature3d[id])

                    p0 = np.dot(self.K0, pc0)
                    p1 = np.dot(self.K1, pc1)
                    u0 = p0[0]/p0[2] + float(np.random.normal(0, 1, 1) * self.feature_noise)
                    v0 = p0[1]/p0[2] + float(np.random.normal(0, 1, 1) * self.feature_noise)
                    u1 = p1[0]/p1[2] + float(np.random.normal(0, 1, 1) * self.feature_noise)
                    v1 = p1[1]/p1[2] + float(np.random.normal(0, 1, 1) * self.feature_noise)
                    if(u0 < 720 and v0 < 480 and u0 > 0 and v0 > 0 
                        and u0 < 720 and v0 < 480 and u0 > 0 and v0 > 0):
                        feature[id] = np.array([u0, v0, u1, v1])
                # print(len(feature))
                if(len(feature) <16):
                    print("error:", i)
                cam_feature.observation = feature
                self.cam_features.append(cam_feature)
            
    def skewSymmetric(self, w):
        w_hat = np.zeros((3,3))
        w_hat[0, 1] = -w[2]
        w_hat[0, 2] = w[1]
        w_hat[1, 0] = w[2]
        w_hat[1, 2] = -w[0]
        w_hat[2, 0] = -w[1]
        w_hat[2, 1] = w[0]
        return w_hat

    def plot_integrate(self, path):
        save_trajectory_path = path + "trajectory.png"
        save_integration_path = path + "integration.png"
        imu_states = []
        R_imu_w_ = np.identity(3)
        R_w_imu_ = np.identity(3)
        q_w_imu_ = np.array([0,0,0,1])
        Vw = np.zeros((3,))
        t_imu_w_ = np.zeros((3,))
        Vw_euler = np.zeros((3,))
        t_imu_w_euler = np.zeros((3,))
        position_gt = []

        for i in range(len(self.imu_dataset)):
            if i == 0:
                quat = [self.imu_dataset[i][2], self.imu_dataset[i][3], self.imu_dataset[i][4],
                        self.imu_dataset[i][1]]
                
                R_imu_w_ = R.from_quat(quat).as_matrix()
                q_w_imu_ = quat
                Vw = np.array([self.imu_dataset[i][8], self.imu_dataset[i][9], self.imu_dataset[i][10]])
                
                euler = euler_from_matrix(R_imu_w_)
                imu_states.append(np.array([self.imu_dataset[i][0], 0, 0, 0, euler[0], euler[1], euler[2]]))
                q = [self.imu_dataset[i][2], self.imu_dataset[i][3], self.imu_dataset[i][4], self.imu_dataset[i][1]]
                euler_gt = euler_from_quaternion(q)
                position_gt.append(np.array([self.imu_dataset[i][5], self.imu_dataset[i][6], self.imu_dataset[i][7], euler_gt[0], euler_gt[1], euler_gt[2]]))

            else:
                dt = self.imu_dataset[i][0] - self.imu_dataset[i-1][0]
                gyro = self.imu_dataset[i][11:14]
                acc = self.imu_dataset[i][14:17]
                
                # euler
                dtheta_half =  gyro * dt /2.0
                dq = np.array([dtheta_half[0], dtheta_half[1], dtheta_half[2], 1])
                dq = dq/np.linalg.norm(dq)
                acc_w = np.dot(R_imu_w_, acc) + np.array([0, 0, self.g_rot])
                R_imu_w_ = np.dot(R_imu_w_, R.from_quat(dq).as_matrix())
                t_imu_w_euler = t_imu_w_euler + Vw_euler * dt + 0.5 * dt * dt * acc_w
                Vw_euler = Vw_euler + acc_w * dt
                euler = euler_from_matrix(R_imu_w_)
                imu_states.append(np.array([self.imu_dataset[i][0], t_imu_w_euler[0], t_imu_w_euler[1], t_imu_w_euler[2], euler[0], euler[1], euler[2]]))

                # rk4
                # gyro_norm = np.linalg.norm(gyro)
                # Omega = np.zeros((4,4))
                # Omega[:3,:3] = -self.skewSymmetric(gyro)
                # Omega[:3,3] = gyro
                # Omega[3,:3] = -gyro


                # if (gyro_norm > 1e-5):
                #     dq_dt = (cos(gyro_norm*dt*0.5)*np.identity(4) +
                #         1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) @ q_w_imu_
                #     dq_dt2 = (cos(gyro_norm*dt*0.25)*np.identity(4) +
                #         1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) @ q_w_imu_
                # else:
                #    dq_dt = (np.identity(4)+0.5*dt*Omega) *cos(gyro_norm*dt*0.5) @ q_w_imu_
                #    dq_dt2 = (np.identity(4)+0.25*dt*Omega) * cos(gyro_norm*dt*0.25) @ q_w_imu_
                 
                # dR_dt_transpose = R.from_quat(dq_dt).as_matrix()
                # dR_dt2_transpose = R.from_quat(dq_dt2).as_matrix()

                # k1_v_dot = R.from_quat(q_w_imu_).as_matrix()@acc + np.array([0, 0, self.g_rot])
                # k1_p_dot = Vw
               
                # k1_v = Vw + k1_v_dot*dt/2
                # k2_v_dot = dR_dt2_transpose@acc + np.array([0, 0, self.g_rot])
                # k2_p_dot = k1_v

                # k2_v = Vw + k2_v_dot*dt/2
                # k3_v_dot = dR_dt2_transpose@acc + np.array([0, 0, self.g_rot])
                # k3_p_dot = k2_v

                # k3_v = Vw + k3_v_dot*dt
                # k4_v_dot = dR_dt_transpose@acc + np.array([0, 0, self.g_rot])
                # k4_p_dot = k3_v

                # q_w_imu_ = dq_dt
                # q_w_imu_ /= np.linalg.norm(q)
                # Vw = Vw + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot)
                # t_imu_w_ = t_imu_w_ + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot)
                # # euler = euler_from_matrix(R.from_quat(q_w_imu_).as_matrix().transpose())
                # euler = euler_from_quaternion(q_w_imu_)
                # imu_states.append(np.array([self.imu_dataset[i][0], t_imu_w_[0], t_imu_w_[1], t_imu_w_[2], euler[0], euler[1], euler[2]]))
   
                q = [self.imu_dataset[i][2], self.imu_dataset[i][3], self.imu_dataset[i][4], self.imu_dataset[i][1]]
                euler_gt = euler_from_quaternion(q)
                position_gt.append(np.array([self.imu_dataset[i][5], self.imu_dataset[i][6], self.imu_dataset[i][7], euler_gt[0], euler_gt[1], euler_gt[2]]))
        
        position_gt = np.array(position_gt)
        imu_states = np.array(imu_states)

        fig1, ax1 = plt.subplots(2,3)
        ax1[0,0].plot(imu_states[:,0],position_gt[:,0],label='real')
        ax1[0,0].plot(imu_states[:,0],imu_states[:,1],label='integration')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(imu_states[:,0],position_gt[:,1])
        ax1[0,1].plot(imu_states[:,0],imu_states[:,2])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(imu_states[:,0],position_gt[:,2])
        ax1[0,2].plot(imu_states[:,0],imu_states[:,3])
        ax1[0,2].set_ylabel('z[m]')
        
        ax1[1,0].plot(imu_states[:,0],position_gt[:,3],label='real')
        ax1[1,0].plot(imu_states[:,0],imu_states[:,4],label='integration')
        ax1[1,0].set_ylabel('x[m]')
        ax1[1,1].plot(imu_states[:,0],position_gt[:,4])
        ax1[1,1].plot(imu_states[:,0],imu_states[:,5])
        ax1[1,1].set_ylabel('y[m]')
        ax1[1,2].plot(imu_states[:,0],position_gt[:,5])
        ax1[1,2].plot(imu_states[:,0],imu_states[:,6])
        ax1[1,2].set_ylabel('z[m]')
        ax1[1,0].legend()
        plt.savefig(save_integration_path, dpi=300)
        
        self.feature3d = np.array(self.feature3d)
        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        ax2.plot3D(position_gt[:,0], position_gt[:,1], position_gt[:,2], c= "b")
        ax2.scatter(self.feature3d[:,0], self.feature3d[:,1], self.feature3d[:,2])
            
        ax2.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax2.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax2.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        plt.legend()
        plt.savefig(save_trajectory_path, dpi=300)
        
        
    
    def save_pose(self, path, bias, gyro_bias):
        imu_path = path + "imu.txt"
        imu_noise_path = path + "imu_noise.txt"
        cam_path = path + "cam.txt"
        feature_path = path + "feature.txt"
        
        self.f = open(imu_path, 'w')
        self.f.write("time qw qx qy qz px py pz vx vy vz gyro_x gyro_y gyro_z accel_x accel_y accel_z")
        self.f.write('\r\n')
        
        for i in range(len(self.imu_dataset)): 
            # self.imu_dataset[i][14] = - self.imu_dataset[i][14]
            # self.imu_dataset[i][15] = - self.imu_dataset[i][15]
            # self.imu_dataset[i][11] = - self.imu_dataset[i][11]
            # self.imu_dataset[i][12] = - self.imu_dataset[i][12]
            data = [str(x) for x in self.imu_dataset[i]]
            self.f.write(' '.join(data))
            self.f.write('\r\n')
        self.f.close()
        print("finish save imu! ")
        
        self.f = open(imu_noise_path, 'w')
        self.f.write("time qw qx qy qz px py pz vx vy vz gyro_x gyro_y gyro_z accel_x accel_y accel_z")
        self.f.write('\r\n')
     
        for i in range(len(self.imu_dataset)):
            self.imu_dataset[i][14:17] = self.imu_dataset[i][14:17] + bias \
                + np.array([random.uniform(-self.noise_w, self.noise_w), random.uniform(-self.noise_w, self.noise_w), \
                            random.uniform(-self.noise_w, self.noise_w)])          
            
            self.imu_dataset[i][11:14] = self.imu_dataset[i][11:14] + gyro_bias             
            data = [str(x) for x in self.imu_dataset[i]]
            self.f.write(' '.join(data))
            self.f.write('\r\n')
        print("finish save imu noise! ")
        
        self.f = open(cam_path, 'w')
        for i in range(len(self.cam_features)):
            cam_feature = self.cam_features[i]
            self.f.write(str(cam_feature.time))
            
            for id in cam_feature.observation:
                feature_uv = cam_feature.observation[id]
                self.f.write(' ')
                self.f.write(str(id))
                self.f.write(' ')
                data = [str(x) for x in feature_uv]
                self.f.write(' '.join(data))
                
            self.f.write('\r\n')
        self.f.close()
        print("finish save cam feature! ")
        
        self.f = open(feature_path, 'w')
        
        for id in range(len(self.feature3d)):
            data = [str(x) for x in self.feature3d[id]]
            self.f.write(' '.join(data))
            self.f.write('\r\n')
        
        
            
        
if __name__ == "__main__":
    
    path = "/home/ldd/bias_esti_ws/src/bias_esti/trajectory/simulation_bias/"
    drone = DroneControlSim()
    drone.run()
    
    drone.generate_cam_feature()
    # drone.plot_states()
    drone.plot_integrate(path)
    bias = [0.02, 0.03, 0.02]
    gyro_bias = [0.01, 0.01, 0.01]
    drone.save_pose(path, bias, gyro_bias)    
    plt.show()