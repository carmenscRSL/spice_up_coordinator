import numpy as np
import rospy
import trimesh
import cv2
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_matrix
import tf2_ros

class poseProcessor:
    '''
    The poseProcessor receives the T_ce pose, the intrinsics and a color-img. Based on that it computes:
    - the four possible grasp poses (in C-Frame)
    - the two possible drop off poses (in C-Frame)
    '''
    def __init__(self, T_ce_msg, K, stamp, color_frame, debug) -> None:
        
        self.debug = debug
        self.K = K
        self.T_ce = self.pose_msg_to_pose_matrix(T_ce_msg)

        self.simple_grasp = True # If True, the alignment-approach is used
        self.correct_rot = True # If True, the grasp pose is aligned with the one from the simulator

        # Get and save mesh props
        mesh_file_path = rospy.get_param("spice_up_coordinator/mesh_file_path")
        self.mesh, self.mesh_props = self.load_mesh(mesh_file_path)
        self.bbox = self.mesh_props["bbox"]
        self.extents = self.mesh_props["extents"]

        self.output_frame = "racetrack"
        self.gripper_frame = "dynaarm_END_EFFECTOR"
        self.alignment_axis = np.array([0, 0, 1])

        # Get shelf h,w,d
        self.shelf_depth = self.extents[0]
        self.shelf_height = self.extents[1] # TODO: CHECK IF STILL TRUE FOR SMALLER KALLAX !! 
        self.shelf_width = self.extents[2] # TODO: CHECK IF STILL TRUE FOR SMALLER KALLAX !! 
        
        # Create T_es -> T_cs for viz (not original T_cs, but the one rotationally aligned with E-frame)
        T_es = np.zeros((4,4))
        T_es[0:3,0:3] = np.identity(3)
        T_es[0:3,3] = np.array([self.shelf_depth/2,self.shelf_height/2,self.shelf_width/2])
        T_es[3,3] = 1

        self.T_cs = self.T_ce @ T_es

        self.T_cdo0,self.T_cdo1 = self.get_drop_off_poses()
        self.T_cg0,self.T_cg1,self.T_cg2,self.T_cg3, = self.get_grasp_poses()

        # Create pose msgs
        T_cg0_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cg0[0:3,3], "spice_up_target0", stamp, safety_margin=0.0)
        T_cg1_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cg1[0:3,3], "spice_up_target1", stamp, safety_margin=0.0)
        T_cg2_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cg2[0:3,3], "spice_up_target2", stamp, safety_margin=0.0)
        T_cg3_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cg3[0:3,3], "spice_up_target3", stamp, safety_margin=0.0)
        
        T_cdo0_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cdo0[0:3,3], "spice_up_dropoff0", stamp, safety_margin=0.0)
        T_cdo1_msg = self.calculate_target_pose(self.gripper_frame, self.alignment_axis, self.T_cdo1[0:3,3], "spice_up_dropoff1", stamp, safety_margin=0.0)

        self.grasp_dict = {
            0: self.T_cg0,
            1: self.T_cg1,
            2: self.T_cg2,
            3: self.T_cg3
        }

        self.grasp_msg_dict = {
            0: T_cg0_msg,
            1: T_cg1_msg,
            2: T_cg2_msg,
            3: T_cg3_msg
        }

        self.drop_off_msg_dict = {
            0: T_cdo0_msg,
            1: T_cdo1_msg
        }

    def calculate_target_pose(
            self, action_frame, target_axis, object_C,
            debug_frame, timestamp, safety_margin=0.0, fix_rotation_axis=None,
            fix_translation_axis=None):
        try: #TODO: do this only once as it's slow af
            tf2_A_C = self.tf_buffer.lookup_transform(
                    action_frame, camera_frame, timestamp, rospy.Duration(tf_timeout))
            tf2_O_A = self.tf_buffer.lookup_transform(
                    self.output_frame, action_frame, timestamp, rospy.Duration(tf_timeout))
            tf2_A_B = self.tf_buffer.lookup_transform(
                    action_frame, "base", timestamp, rospy.Duration(tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            print("[FaceDetectionNode]: TF lookup failed.", error)
            return False

        # Convert to transformation matrix
        q_A_C = np.quaternion(
                tf2_A_C.transform.rotation.w,
                tf2_A_C.transform.rotation.x,
                tf2_A_C.transform.rotation.y,
                tf2_A_C.transform.rotation.z)
        T_A_C = np.eye(4)
        T_A_C[0:3, 0:3] = quaternion.as_rotation_matrix(q_A_C)
        T_A_C[0, 3] = tf2_A_C.transform.translation.x 
        T_A_C[1, 3] = tf2_A_C.transform.translation.y
        T_A_C[2, 3] = tf2_A_C.transform.translation.z

        q_O_A = np.quaternion(
                tf2_O_A.transform.rotation.w,
                tf2_O_A.transform.rotation.x,
                tf2_O_A.transform.rotation.y,
                tf2_O_A.transform.rotation.z)
        T_O_A = np.eye(4)
        T_O_A[0:3, 0:3] = quaternion.as_rotation_matrix(q_O_A)
        T_O_A[0, 3] = tf2_O_A.transform.translation.x
        T_O_A[1, 3] = tf2_O_A.transform.translation.y
        T_O_A[2, 3] = tf2_O_A.transform.translation.z

        object_A = T_A_C @ object_C.T
        object_O = T_O_A @ object_A.T

        # Vector from robot to object in robot frame is the position
        # of the object. We will normalize later.
        base_offset = np.zeros(3)
        base_offset[0] = tf2_A_B.transform.translation.x
        base_offset[1] = tf2_A_B.transform.translation.y
        base_offset[2] = tf2_A_B.transform.translation.z
        print(base_offset)
        vector_A = object_A[:3].copy() - base_offset
        print(vector_A, object_A[:3])
        # We want to align the given frame axis to the vector going from
        # the action frame and the detected object.
        vector_X = np.array(target_axis, dtype=np.double)

        if fix_rotation_axis is not None:
            # Do not rotate along this axis but keep it fixed in the
            # with respect to the action_frame
            vector_A[fix_rotation_axis] = 0
            vector_X[fix_rotation_axis] = 0

        # Normalize the vectors
        vector_A /= np.linalg.norm(vector_A)
        vector_X /= np.linalg.norm(vector_X)

        # Get how much robot must rotate to face the object
        q_A_AT = get_quaternion_from_vectors(vector_X, vector_A)
        q_O_AT = q_O_A * q_A_AT

        base_target_A = object_A.copy()

        if fix_translation_axis is not None:
            # Do not translate this axis but keep the position offset constant
            base_target_A[fix_translation_axis] = 0

        if abs(safety_margin) > 0.0:
            # Move the robot backwards the requested distance
            base_target_A[:3] -= vector_A * safety_margin 

        # Convert target position into target output frame
        base_target_O = T_O_A @ base_target_A.T

        # Normalize the quaternion, otherwise ROS doesn't handle it
        q_O_AT = q_O_AT.normalized()

        print("[FaceDetectionNode]: For target frame %s" % action_frame,
                "found quaternion", q_O_AT, "and translation", base_target_O)

        if debug:
            self.visualize_frame(q_O_AT, base_target_O, self.output_frame,
                    debug_frame)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.output_frame
        pose_msg.pose.position.x = base_target_O[0]
        pose_msg.pose.position.y = base_target_O[1]
        pose_msg.pose.position.z = base_target_O[2]
        pose_msg.pose.orientation.x = q_O_AT.x
        pose_msg.pose.orientation.y = q_O_AT.y
        pose_msg.pose.orientation.z = q_O_AT.z
        pose_msg.pose.orientation.w = q_O_AT.w

        return pose_msg


    def get_grasp_poses(self):

        '''
        C-frame: camera frame
        E-frame: 
            position: bottom right front corner of shelf
            rotation: x: depth of shelf, y: width of shelf, z: height of shelf
        M-frame (M0,M1,M2,M3)
            position: center of contact circle between spice-bottle-i and shelf
            rotation: same as E-Frame
        B-frame: anymal base frame (so far only faked in this function)

        _____
        |2|3|
        -----
        |0|1|
        -----
        '''

        # CONSTRUCT GRASPING POSES in E-frame
        z_off = 0.05
        EP0_r_E = np.array([0.040,0.285,0.392+z_off,1.0])
        EP1_r_E = np.array([0.040,0.125,0.392+z_off,1.0])
        EP2_r_E = np.array([0.035,0.280,0.042+z_off,1.0])
        EP3_r_E = np.array([0.035,0.130,0.042+z_off,1.0])

        # Construct transforms from E t0 m0,m1,m2,m3 (bottles middle points))
        T_em0 = np.eye(4)
        T_em0[0:4,3] = EM0_r_E

        T_em1 = np.eye(4)
        T_em1[0:4,3] = EM1_r_E

        T_em2 = np.eye(4)
        T_em2[0:4,3] = EM2_r_E

        T_em3 = np.eye(4)
        T_em3[0:4,3] = EM3_r_E

        # Bring into camera frame
        T_cg0 = self.T_ce @ T_em0  
        T_cg1 = self.T_ce @ T_em1  
        T_cg2 = self.T_ce @ T_em2  
        T_cg3 = self.T_ce @ T_em3

        return T_cg0,T_cg1,T_cg2,T_cg3  

    def get_drop_off_poses(self):
        DO0_E = np.array([self.shelf_depth/2,self.shelf_width*0.75,self.shelf_height,1.0]) # Dropoff0 position in E-frame
        DO1_E = np.array([self.shelf_depth/2,self.shelf_width*0.25,self.shelf_height,1.0]) # Dropoff1 position in E-frame
        
        T_cdo0 = np.eye(4) # Dropoff0 pose in C-frame
        T_cdo0[:,3] = self.T_ce @ DO0_E
    
        T_cdo1 = np.eye(4) # Dropoff1 pose in C-frame
        T_cdo1[:,3] = self.T_ce @ DO1_E

        return T_cdo0,T_cdo1

    # Utils  -------------------------------------

    def pose_msg_to_pose_matrix(self,pose_msg):

        T = np.zeros((4,4))
        T[0,3] = pose_msg.pose.position.x
        T[1,3] = pose_msg.pose.position.y
        T[2,3] = pose_msg.pose.position.z
        
        quat = [0,0,0,0]
        quat[0] = pose_msg.pose.orientation.x 
        quat[1] = pose_msg.pose.orientation.y
        quat[2] = pose_msg.pose.orientation.z 
        quat[3] = pose_msg.pose.orientation.w

        T[0:3,0:3] = quaternion_matrix(quat)[0:3,0:3] #https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py line 1174
        T[3,3] = 1

        return T

    def load_mesh(self, mesh_file):
        mesh = trimesh.load(mesh_file, force="mesh")
        mesh_props = dict()
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        mesh_props["to_origin"] = to_origin
        mesh_props["bbox"] = bbox
        mesh_props["extents"] = extents
        return mesh, mesh_props
