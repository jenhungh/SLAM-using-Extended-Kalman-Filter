import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.linalg as la
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def drawCovEllipse(c, cov, setting):
    """Draw the Covariance ellipse given the mean and covariance

    :c: Ellipse center
    :cov: Covariance matrix for the state
    :returns: None

    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2*np.pi, np.pi/50)
    rot = []
    for i in range(100):
        rect = (np.array([3*np.sqrt(a)*np.cos(phi[i]), 3*np.sqrt(b)*np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + c)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=setting, linewidth=0.75)


def drawTrajAndMap(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    drawCovEllipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            drawCovEllipse(X[3 + k*2:3 + k*2+2], P[3 + k*2:3 + 2*k + 2, 3 + 2*k:3 + 2*k + 2], 'r')
    else:
        for k in range(6):
            drawCovEllipse(X[3 + k*2:3 + k*2+2], P[3 + 2*k:3 + 2*k + 2, 3 + 2*k:3 + 2*k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def drawTrajPre(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    drawCovEllipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)

def main():
    """Main function for EKF

    :arg1: TODO
    :returns: TODO

    """
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.1
    sig_r = 0.16

    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file
    data_file = open("../../data/data.txt", 'r')

    # Read the first measurement data
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = arr[:, None]
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = (np.array([0, 0, 0]))[:, None]
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    # TODO: Setup the initial landmark estimates landmark[] and covariance matrix landmark_cov[]
    # Hint: use initial pose with uncertainty and first measurement

    ##############################################################
    ################## Write your code here ######################

    # Compute the number of landmarks
    k = int(len(measure)/2)
    # print(f"number of landmarks:{k}")

    # Compute the initial landmark estimates
    landmark = []
    for b,r in zip(range(0,len(measure),2),range(1,len(measure),2)):
        l_x = pose[0]+measure[r]*np.cos(pose[2]+measure[b]) 
        l_y = pose[1]+measure[r]*np.sin(pose[2]+measure[b])
        landmark.append([l_x, l_y])
    landmark = (np.asarray(landmark)).reshape(-1,1)
    # print(f"landmark =\n{landmark}, shape = {landmark.shape}")
    
    # Compute the initial landmark covariance matrix
    landmark_cov = np.array([])
    for b,r in zip(range(0,len(measure),2),range(1,len(measure),2)):
        G_p = np.array([[1, 0, -(measure[r])*np.sin(pose[2]+measure[b])],
                        [0, 1, (measure[r])*np.cos(pose[2]+measure[b])]],dtype=object)
        L_noise = np.array([[-(measure[r])*np.sin(pose[2]+measure[b]), np.cos(pose[2]+measure[b])],
                            [(measure[r])*np.cos(pose[2]+measure[b]), np.sin(pose[2]+measure[b])]],dtype=object).reshape((2,2))
        landmark_cov_i = G_p.dot(pose_cov).dot(G_p.T) + L_noise.dot(measure_cov).dot(L_noise.T)
        landmark_cov = la.block_diag(landmark_cov, landmark_cov_i)
    landmark_cov = np.delete(landmark_cov,0,0)
    landmark_cov = np.asarray(landmark_cov,dtype=np.float64)
    # print(f"landmark_cov =\n{landmark_cov}, shape = {landmark_cov.shape}")

    ##############################################################

    # Setup state vector x with pose and landmark vector
    X = np.vstack((pose, landmark))

    # Setup covariance matrix P with pose and landmark covariance
    P = np.block([[pose_cov,           np.zeros((3, 2*k))],
                  [np.zeros((2*k, 3)),       landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    drawTrajAndMap(X, last_X, P, 0)

    # Read file sequentially for controls
    # and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])
        if arr.shape[0] == 2:
            d, alpha = arr[0], arr[1]
            control = (np.array([d, alpha]))[:, None]

            # TODO: Predict step
            # (Notice: predict state x_pre[] and covariance P_pre[] using input control data and control_cov[])

            ##############################################################
            ################## Write your code here ######################

            # Compute the predict state
            F_x = np.concatenate((np.identity(3),np.zeros((3,12))),axis=1)
            # print(F_x)
            odometry = np.array([(control[0])*np.cos(float(last_X[2])),
                                 (control[0])*np.sin(float(last_X[2])),
                                 control[1]],dtype=object).reshape(-1,1)
            X_pre = X + F_x.T.dot(odometry)
            X_pre = np.asarray(X_pre,dtype=np.float64)
            # print(f"X_pre =\n{X_pre}, shape = {X_pre.shape}")
            
            # Compute the predict state covariance matrix  
            odometry_pose_derivative = np.array([[0, 0, -control[0]*np.sin(last_X[2])],
                                                 [0, 0, control[0]*np.cos(last_X[2])],
                                                 [0, 0, 0]],dtype=object)
            G_t = np.identity(15) + F_x.T.dot(odometry_pose_derivative).dot(F_x)
            L_t = np.array([[np.cos(float(last_X[2])), -np.sin(float(last_X[2])), 0],[np.sin(float(last_X[2])), np.cos(float(last_X[2])), 0],[0, 0, 1]],dtype=object)
            # print(L_t)
            R_t = L_t.dot(control_cov).dot(L_t.T)
            P_pre = G_t.dot(P).dot(G_t.T) + F_x.T.dot(R_t).dot(F_x)
            P_pre = np.asarray(P_pre,dtype=np.float64)
            # print(f"P_pre =\n{P_pre}, shape = {P_pre.shape}")

            ##############################################################


            # Draw predicted state X_pre and Covariance P_pre
            drawTrajPre(X_pre, P_pre)

        # Read the measurement data
        else:
            measure = (arr)[:, None]

            # TODO: Correction step
            # (Notice: Update state X[] and covariance P[] using the input measurement data and measurement_cov[])

            ##############################################################
            ################## Write your code here ######################

            # Meaurement Update
            for x,y in zip(range(3,len(X),2),range(4,len(X),2)):
                F_up = np.concatenate((np.identity(3),np.zeros((3,12))),axis=1)
                F_down = np.concatenate((np.zeros((2,3)),np.zeros((2,x-3)),np.identity(2),np.zeros((2,13-x))),axis=1)
                F = np.concatenate((F_up,F_down),axis=0)
                # print(f"F =\n{F}")

                # Compute the Jacobian H_t
                q = float((X_pre[x]-X_pre[0])**2+(X_pre[y]-X_pre[1])**2)
                dbeta_dx = (X_pre[y]-X_pre[1])/q
                dbeta_dy = -(X_pre[x]-X_pre[0])/q
                dbeta_dtheta = -1
                dbeta_dlx = -(X_pre[y]-X_pre[1])/q
                dbeta_dly = (X_pre[x]-X_pre[0])/q
                dr_dx = -(X_pre[x]-X_pre[0])/np.sqrt(q)
                dr_dy = -(X_pre[y]-X_pre[1])/np.sqrt(q)
                dr_dtheta = 0
                dr_dlx = (X_pre[x]-X_pre[0])/np.sqrt(q)
                dr_dly = (X_pre[y]-X_pre[1])/np.sqrt(q)
                H_t = np.array([[dbeta_dx,dbeta_dy,dbeta_dtheta,dbeta_dlx,dbeta_dly],[dr_dx,dr_dy,dr_dtheta,dr_dlx,dr_dly]],dtype=object).reshape(2,5)
                H_t = np.asarray(H_t,dtype=np.float64)
                # print(f"H_t =\n{H_t}, shape = {H_t.shape}")
                H_T = H_t.dot(F)
                # print(f"H_T =\n{H_T}, shape = {H_T.shape}")

                # Compute Kalman gain K_t
                S = H_T.dot(P_pre).dot(H_T.T) + measure_cov
                K_t = P_pre.dot(H_T.T).dot(np.linalg.inv(S))
                K_t = np.asarray(K_t,dtype=np.float64)
                # print(f"K_t =\n{K_t}, shape = {K_t.shape}")
                
                # Compute the meaurement prediction 
                beta = np.arctan2((X_pre[y]-X_pre[1]),(X_pre[x]-X_pre[0]))-X_pre[2]
                beta_wrap = np.arctan2(np.sin(beta),np.cos(beta))
                measure_pred = np.array([[beta_wrap],
                                         [np.sqrt((X_pre[x]-X_pre[0])**2+(X_pre[y]-X_pre[1])**2)]])
                measure_pred = np.asarray(measure_pred,dtype=np.float64)
                measure_pred = measure_pred.reshape(-1,1)
                # print(f"measure_pred =\n{measure_pred}, shape = {measure_pred.shape}")

                single_measure = measure[x-3:x-1]
                # print(f"single_measure =\n{single_measure}, shape = {single_measure.shape}")
                
                # Measurement Update
                X_pre += K_t.dot(single_measure-measure_pred)
                P_pre = (np.identity(15)-K_t.dot(H_T)).dot(P_pre)
            
            # Update the state and covariance each time step
            X = X_pre 
            P = P_pre
            print(f"State X =\n{X}, shape = {X.shape}")
            print(f"Covariance P =\n{P}, shape = {P.shape}")

            ##############################################################

            drawTrajAndMap(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks

    ##############################################################
    ################## Write your code here ######################
    
            # Plot the true landmark position
            true_landmarks = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12]).reshape(-1,1)
            for i,j in zip(range(0,13,2), range(1,13,2)):
                plt.plot(true_landmarks[i], true_landmarks[j],'ko')
            plt.show()
            
            # Compute the Euclidean and Mahalanobis distance  
            Euclidean = []
            Mahalanobis = []
            for x,y in zip(range(3,15,2), range(4,15,2)):
                Euclidean_d = np.sqrt((X[x]-true_landmarks[x-3])**2+(X[y]-true_landmarks[y-3])**2)
                Euclidean.append(float(Euclidean_d))
                Estimate = np.array([X[x], X[y]])
                Truth = np.array([true_landmarks[x-3], true_landmarks[y-3]])
                Cov = P[x:x+2,x:x+2]
                Mahalanobis_d = np.sqrt((Estimate.T-Truth.T).dot(Cov).dot(Estimate-Truth))
                Mahalanobis.append(float(Mahalanobis_d))
            for i in range(6):
                print(f"The {i+1} landmark")
                print(f"Eucledian = {Euclidean[i]}")
                print(f"Mahalanobis = {Mahalanobis[i]}")

    ##############################################################



if __name__ == "__main__":
    main()
