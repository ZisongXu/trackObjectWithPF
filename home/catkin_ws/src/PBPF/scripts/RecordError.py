#!/usr/bin/python3
import signal


compute_error_flag = False
# panda data frame to record the error and to compare them
# pos
if compute_error_flag == True:
    boss_obse_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_PBPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_CVPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    # ang
    boss_obse_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_PBPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_CVPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    
    # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    boss_opti_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_opti_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_opti_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_opti_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_opti_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_opti_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_opti_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])
    boss_estPB_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_estPB_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_estPB_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_estPB_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_estPB_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_estPB_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_estPB_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])
    boss_estDO_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_estDO_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_estDO_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_estDO_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_estDO_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_estDO_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_estDO_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])


def signal_handler(sig, frame):
    # write the error file
    # if rospy.is_shutdown():
    if update_style_flag == "pose":
        if task_flag == "1":
            file_name_obse_pos = 'pose_scene1_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene1_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene1_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene1_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene1_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene1_CVPF_err_ang.csv'
        elif task_flag == "2":
            file_name_obse_pos = 'pose_scene2_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene2_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene2_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene2_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene2_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene2_CVPF_err_ang.csv'
        elif task_flag == "3":
            file_name_obse_pos = 'pose_scene3_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene3_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene3_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene3_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene3_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene3_CVPF_err_ang.csv'
        elif task_flag == "4":
            file_name_obse_pos = 'pose_scene4_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene4_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene4_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene4_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene4_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene4_CVPF_err_ang.csv'
    elif update_style_flag == "time":
        if task_flag == "1":
            file_name_obse_pos = 'time_scene1_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene1_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene1_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene1_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene1_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene1_CVPF_err_ang.csv'
        elif task_flag == "2":
            file_name_obse_pos = 'time_scene2_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene2_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene2_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene2_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene2_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene2_CVPF_err_ang.csv'
        elif task_flag == "3":
            file_name_obse_pos = 'time_scene3_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene3_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene3_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene3_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene3_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene3_CVPF_err_ang.csv'
        elif task_flag == "4":
            file_name_obse_pos = 'time_scene4_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene4_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene4_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene4_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene4_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene4_CVPF_err_ang.csv'
    # boss_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_pos,index=0,header=0,mode='a')
    # boss_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_ang,index=0,header=0,mode='a')
    if run_PBPF_flag == True:
        boss_obse_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_obse_pos,index=0,header=0,mode='a')
        boss_obse_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_obse_ang,index=0,header=0,mode='a')
        print("write obser file")
        boss_PBPF_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_PBPF_pos,index=0,header=0,mode='a')
        boss_PBPF_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_PBPF_ang,index=0,header=0,mode='a')
        print("write PBPF file")
        print("PB: Update frequency is: " + str(flag_update_num_PB))
        print("max time:", max(PBPF_time_cosuming_list))
    if run_CVPF_flag == True:
        boss_CVPF_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_CVPF_pos,index=0,header=0,mode='a')
        boss_CVPF_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_CVPF_ang,index=0,header=0,mode='a')
        print("write CVPF file")
        print("CV: Update frequency is: " + str(flag_update_num_CV))
    print("file_time:", file_time)
    # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    if write_opti_pose_flag == True:
        print("write_opti_pos")
        boss_opti_pos_x_df.to_csv('opti_pos_x.csv')
        boss_opti_pos_y_df.to_csv('opti_pos_y.csv')
        boss_opti_pos_z_df.to_csv('opti_pos_z.csv')
        boss_opti_ori_x_df.to_csv('opti_ori_x.csv')
        boss_opti_ori_y_df.to_csv('opti_ori_y.csv')
        boss_opti_ori_z_df.to_csv('opti_ori_z.csv')
        boss_opti_ori_w_df.to_csv('opti_ori_w.csv')
    if write_estPB_pose_flag == True:
        print("write_opti_pos")
        boss_estPB_pos_x_df.to_csv('estPB_pos_x.csv')
        boss_estPB_pos_y_df.to_csv('estPB_pos_y.csv')
        boss_estPB_pos_z_df.to_csv('estPB_pos_z.csv')
        boss_estPB_ori_x_df.to_csv('estPB_ori_x.csv')
        boss_estPB_ori_y_df.to_csv('estPB_ori_y.csv')
        boss_estPB_ori_z_df.to_csv('estPB_ori_z.csv')
        boss_estPB_ori_w_df.to_csv('estPB_ori_w.csv')
    if write_estDO_pose_flag == True:
        print("write_estDO_pos")
        boss_estPB_pos_x_df.to_csv('estDO_pos_x.csv')
        boss_estPB_pos_y_df.to_csv('estDO_pos_y.csv')
        boss_estPB_pos_z_df.to_csv('estDO_pos_z.csv')
        boss_estPB_ori_x_df.to_csv('estDO_ori_x.csv')
        boss_estPB_ori_y_df.to_csv('estDO_ori_y.csv')
        boss_estPB_ori_z_df.to_csv('estDO_ori_z.csv')
        boss_estPB_ori_w_df.to_csv('estDO_ori_w.csv')
    sys.exit()

signal.signal(signal.SIGINT, signal_handler) # interrupt judgment



            if compute_error_flag == True:
                opti_from_pre_time = time.time()
                boss_opti_pos_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[0], 'opti']
                boss_opti_pos_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[1], 'opti']
                boss_opti_pos_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[2], 'opti']
                boss_opti_ori_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[0], 'opti']
                boss_opti_ori_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[1], 'opti']
                boss_opti_ori_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[2], 'opti']
                boss_opti_ori_w_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[3], 'opti']
                opti_form_previous = opti_form_previous + 1
                
    # compute error
    if optitrack_working_flag == True:
        err_opti_obse_pos = compute_pos_err_bt_2_points(pw_T_obj_opti_pos, pw_T_obj_obse_pos)
        err_opti_obse_ang = compute_ang_err_bt_2_points(pw_T_obj_opti_ori, pw_T_obj_obse_ori)
        err_opti_obse_ang = angle_correction(err_opti_obse_ang)
    elif optitrack_working_flag == False:
        err_opti_obse_pos = compute_pos_err_bt_2_points(ros_listener.fake_opti_pos, pw_T_obj_obse_pos)
        err_opti_obse_ang = compute_ang_err_bt_2_points(ros_listener.fake_opti_ori, pw_T_obj_obse_ori)
        err_opti_obse_ang = angle_correction(err_opti_obse_ang)
# when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    if compute_error_flag == True:
        estPB_from_pre_time = time.time()
        boss_estPB_pos_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[0], 'estPB']
        boss_estPB_pos_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[1], 'estPB']
        boss_estPB_pos_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[2], 'estPB']
        boss_estPB_ori_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[0], 'estPB']
        boss_estPB_ori_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[1], 'estPB']
        boss_estPB_ori_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[2], 'estPB']
        boss_estPB_ori_w_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[3], 'estPB']
        estPB_form_previous = estPB_form_previous + 1
        boss_estDO_pos_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_pos[0], 'estDO']
        boss_estDO_pos_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_pos[1], 'estDO']
        boss_estDO_pos_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_pos[2], 'estDO']
        boss_estDO_ori_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_ori[0], 'estDO']
        boss_estDO_ori_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_ori[1], 'estDO']
        boss_estDO_ori_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_ori[2], 'estDO']
        boss_estDO_ori_w_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_obse_ori[3], 'estDO']
        estDO_form_previous = estDO_form_previous + 1
    # publish pose
    if publish_PBPF_pose_flag == True:
        pub = rospy.Publisher('PBPF_pose', PoseStamped, queue_size = 1)
        pose_PBPF = PoseStamped()
        pose_PBPF.pose.position.x = estimated_object_pos[0]
        pose_PBPF.pose.position.y = estimated_object_pos[1]
        pose_PBPF.pose.position.z = estimated_object_pos[2]
        pose_PBPF.pose.orientation.x = estimated_object_ori[0]
        pose_PBPF.pose.orientation.y = estimated_object_ori[1]
        pose_PBPF.pose.orientation.z = estimated_object_ori[2]
        pose_PBPF.pose.orientation.w = estimated_object_ori[3]
        pub.publish(pose_PBPF)
        # rospy.loginfo(pose_PBPF)
    if publish_obse_pose_flag == True:
        pub_obse = rospy.Publisher('OBSE_pose', PoseStamped, queue_size = 1)
        pose_obse = PoseStamped()
        pose_obse.pose.position.x = pw_T_obj_obse_pos[0]
        pose_obse.pose.position.y = pw_T_obj_obse_pos[1]
        pose_obse.pose.position.z = pw_T_obj_obse_pos[2]
        pose_obse.pose.orientation.x = pw_T_obj_obse_ori[0]
        pose_obse.pose.orientation.y = pw_T_obj_obse_ori[1]
        pose_obse.pose.orientation.z = pw_T_obj_obse_ori[2]
        pose_obse.pose.orientation.w = pw_T_obj_obse_ori[3]
        # print(pose_obse)
        pub_obse.publish(pose_obse)
        # rospy.loginfo(pose_obse)
        
        if optitrack_working_flag == True:
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_Opti_pose_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = pw_T_obj_opti_pos[0]
            pose_opti.pose.position.y = pw_T_obj_opti_pos[1]
            pose_opti.pose.position.z = pw_T_obj_opti_pos[2]
            pose_opti.pose.orientation.x = pw_T_obj_opti_ori[0]
            pose_opti.pose.orientation.y = pw_T_obj_opti_ori[1]
            pose_opti.pose.orientation.z = pw_T_obj_opti_ori[2]
            pose_opti.pose.orientation.w = pw_T_obj_opti_ori[3]
            pub_opti.publish(pose_opti)
            
            # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
            if compute_error_flag == True:
                opti_from_pre_time = time.time()
                boss_opti_pos_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[0], 'opti']
                boss_opti_pos_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[1], 'opti']
                boss_opti_pos_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[2], 'opti']
                boss_opti_ori_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[0], 'opti']
                boss_opti_ori_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[1], 'opti']
                boss_opti_ori_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[2], 'opti']
                boss_opti_ori_w_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[3], 'opti']
                opti_form_previous = opti_form_previous + 1
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_opti_pose_for_inter_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = pw_T_obj_opti_pos[0]
            pose_opti.pose.position.y = pw_T_obj_opti_pos[1]
            pose_opti.pose.position.z = pw_T_obj_opti_pos[2]
            pose_opti.pose.orientation.x = pw_T_obj_opti_ori[0]
            pose_opti.pose.orientation.y = pw_T_obj_opti_ori[1]
            pose_opti.pose.orientation.z = pw_T_obj_opti_ori[2]
            pose_opti.pose.orientation.w = pw_T_obj_opti_ori[3]
            pub_opti.publish(pose_opti)
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_obse_err_ang_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            boss_err_pos_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_err_ang_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            flag_record = flag_record + 1
                            flag_record_obse = flag_record_obse + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1
                    if first_write_flag == 0 and compute_error_flag == True:
                        # record the error
                        t_begin = time.time()
                        t_before_record = time.time()
                        boss_obse_err_pos_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                        boss_obse_err_ang_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                        boss_err_pos_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                        boss_err_ang_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                        flag_record = flag_record + 1
                        flag_record_obse = flag_record_obse + 1
                        boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                        boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                        boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                        boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                        flag_record = flag_record + 1
                        flag_record_PBPF = flag_record_PBPF + 1
                        boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                        boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                        boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                        boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                        flag_record = flag_record + 1
                        flag_record_CVPF = flag_record_CVPF + 1
                        first_write_flag = 1
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_obse_err_ang_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            boss_err_pos_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_err_ang_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            flag_record = flag_record + 1
                            flag_record_obse = flag_record_obse + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_obse_err_ang_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            boss_err_pos_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
                            boss_err_ang_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
                            flag_record = flag_record + 1
                            flag_record_obse = flag_record_obse + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1


# PBPF:                   
        # compute error and write down to the file
        err_opti_obse_pos = compute_pos_err_bt_2_points(nois_obj_pos_cur,opti_obj_pos_cur)
        err_opti_obse_ang = compute_ang_err_bt_2_points(nois_obj_ori_cur,opti_obj_ori_cur)
        err_opti_obse_ang = angle_correction(err_opti_obse_ang)
        err_opti_PBPF_pos = compute_pos_err_bt_2_points(estimated_object_pos,opti_obj_pos_cur)
        err_opti_PBPF_ang = compute_ang_err_bt_2_points(estimated_object_ori,opti_obj_ori_cur)
        err_opti_PBPF_ang = angle_correction(err_opti_PBPF_ang)
        if publish_PBPF_pose_flag == True:
            pub = rospy.Publisher('PBPF_pose', PoseStamped, queue_size = 1)
            pose_PBPF = PoseStamped()
            pose_PBPF.pose.position.x = estimated_object_pos[0]
            pose_PBPF.pose.position.y = estimated_object_pos[1]
            pose_PBPF.pose.position.z = estimated_object_pos[2]
            pose_PBPF.pose.orientation.x = estimated_object_ori[0]
            pose_PBPF.pose.orientation.y = estimated_object_ori[1]
            pose_PBPF.pose.orientation.z = estimated_object_ori[2]
            pose_PBPF.pose.orientation.w = estimated_object_ori[3]
            pub.publish(pose_PBPF)
        if publish_obse_pose_flag == True:
            pub_obse = rospy.Publisher('obse_pose', PoseStamped, queue_size = 1)
            pose_obse = PoseStamped()
            pose_obse.pose.position.x = nois_obj_pos_cur[0]
            pose_obse.pose.position.y = nois_obj_pos_cur[1]
            pose_obse.pose.position.z = nois_obj_pos_cur[2]
            pose_obse.pose.orientation.x = nois_obj_ori_cur[0]
            pose_obse.pose.orientation.y = nois_obj_ori_cur[1]
            pose_obse.pose.orientation.z = nois_obj_ori_cur[2]
            pose_obse.pose.orientation.w = nois_obj_ori_cur[3]
            pub_obse.publish(pose_obse)
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_Opti_pose_flag == True and optitrack_working_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = opti_obj_pos_cur[0]
            pose_opti.pose.position.y = opti_obj_pos_cur[1]
            pose_opti.pose.position.z = opti_obj_pos_cur[2]
            pose_opti.pose.orientation.x = opti_obj_ori_cur[0]
            pose_opti.pose.orientation.y = opti_obj_ori_cur[1]
            pose_opti.pose.orientation.z = opti_obj_ori_cur[2]
            pose_opti.pose.orientation.w = opti_obj_ori_cur[3]
            pub_opti.publish(pose_opti)
        if compute_error_flag == True:
            t_before_record = time.time()
            boss_obse_err_pos_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
            boss_obse_err_ang_df.loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
            boss_err_pos_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse']
            boss_err_ang_df.loc[flag_record] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse']
            flag_record = flag_record + 1
            flag_record_obse = flag_record_obse + 1
            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_pos, 'PBPF']
            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_ang, 'PBPF']
            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_pos, 'PBPF']
            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_ang, 'PBPF']
            flag_record = flag_record + 1
            flag_record_PBPF = flag_record_PBPF + 1
            estPB_from_pre_time = time.time()
            boss_estPB_pos_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[0], 'estPB']
            boss_estPB_pos_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[1], 'estPB']
            boss_estPB_pos_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[2], 'estPB']
            boss_estPB_ori_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[0], 'estPB']
            boss_estPB_ori_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[1], 'estPB']
            boss_estPB_ori_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[2], 'estPB']
            boss_estPB_ori_w_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[3], 'estPB']
            estPB_form_previous = estPB_form_previous + 1
            boss_estDO_pos_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[0], 'estDO']
            boss_estDO_pos_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[1], 'estDO']
            boss_estDO_pos_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[2], 'estDO']
            boss_estDO_ori_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[0], 'estDO']
            boss_estDO_ori_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[1], 'estDO']
            boss_estDO_ori_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[2], 'estDO']
            boss_estDO_ori_w_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[3], 'estDO']
            estDO_form_previous = estDO_form_previous + 1

# CVPF:   
        # compute error and write down to the file
        err_opti_CVPF_pos = compute_pos_err_bt_2_points(estimated_object_pos_CV, opti_obj_pos_cur)
        err_opti_CVPF_ang = compute_ang_err_bt_2_points(estimated_object_ori_CV, opti_obj_ori_cur)
        err_opti_CVPF_ang = angle_correction(err_opti_CVPF_ang)
        if compute_error_flag == True:
            t_before_record = time.time()
            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_pos, 'CVPF']
            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_ang, 'CVPF']
            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_pos, 'CVPF']
            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_ang, 'CVPF']
            flag_record = flag_record + 1
            flag_record_CVPF = flag_record_CVPF + 1
