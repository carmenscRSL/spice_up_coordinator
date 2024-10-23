#!/usr/bin/env python3

import numpy as np
import cv2
# Ros 
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import actionlib
# Ros msgs
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
# Custom msgs
from cybathlon_bt_msgs.msg import SpiceUpBottlePickAction,SpiceUpBottlePickResult
from spice_up_coordinator.srv._GetSpiceName import GetSpiceName, GetSpiceNameRequest
from spice_up_coordinator.srv._EstimatePose import EstimatePose, EstimatePoseRequest
# Files
from pose_processor import poseProcessor

class spiceUpCoordinator:
    def __init__(self):
        '''
        The spiceUpCoordinator coordinates the requests and responses between the index_finder_server, spice_name_server and pose_est_server during the spice-up-task.
        A flow chart depecting the information flow can be found on the github repo TODO
        '''
        rospy.init_node('spice_up_action_server')

        self.load_params()

        # Set up callbacks
        color_img_sub = message_filters.Subscriber(self.color_topic_name,Image, queue_size=1)
        depth_img_sub = message_filters.Subscriber(self.depth_topic_name,Image, queue_size=1)
        ts = message_filters.TimeSynchronizer([color_img_sub, depth_img_sub], 10)
        ts.registerCallback(self.synch_image_callback)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Setup publishers
        self.shutdown_pub = rospy.Publisher("/shutdown_spice_up",Bool)
        self.pose_debug_pub = rospy.Publisher("/debug_pose", Image, queue_size=1)
        self.shutdown_sub = rospy.Subscriber("/shutdown_spice_up",Bool, self.shutdown_cb)

        # Start action service
        self.action_server = actionlib.SimpleActionServer("spice_up_action_server", SpiceUpBottlePickAction, execute_cb=self.service_cb, auto_start = False)
        self.action_server.start()

        print("[spiceUpCoordinator] : Waiting for get_spice_name_service...")
        rospy.wait_for_service('get_spice_name_service')
        print("[spiceUpCoordinator] : get_spice_name_service found")

        print("[spiceUpCoordinator] : Waiting for estimate_pose_service...")
        rospy.wait_for_service('estimate_pose_service')
        print("[spiceUpCoordinator] : estimate_pose_service found")

        print("[spiceUpCoordinator] : "+str("Initialized"))
        
    def service_cb(self, goal): # Goal of type: SpiceUpBottlePickGoal
        
        print("[spiceUpCoordinator] : Received action goal")
        br = tf.TransformBroadcaster()
        
        result = SpiceUpBottlePickResult()
        result.ee_pickup_target = PoseStamped()
        result.ee_intermediary_target = PoseStamped()
        result.ee_dropoff_target = PoseStamped()

        if goal.activation:       
            # If this is the first action request: request pose and upon receiving response generate grasp and dropoff poses
            if self.drop_off_index == 0: 
                pose_request = EstimatePoseRequest(self.last_image_color_msg,self.last_image_depth_msg)
                self.stamp = self.last_image_color_msg.header.stamp
                pose_service_handle = rospy.ServiceProxy('estimate_pose_service', EstimatePose)
                print("[spiceUpCoordinator] : Requesting shelf pose")
                pose_service_response = pose_service_handle(pose_request)
                T_ce_msg = pose_service_response.T_ce
                # self.mask_msg = pose_service_response.mask
                # self.mask_has_five_contours = pose_service_response.has_five_contours
                print("[spiceUpCoordinator] : Received shelf pose: "+str(T_ce_msg))
                self.poseProcessor = poseProcessor(T_ce_msg,self.K,self.stamp,self.last_image_color,self.debug,self.tf_buffer,self.listener)
                print("[spiceUpCoordinator] : Generated grasp and dropoff poses")

            # Send request for spice_name to spice_name_server
            spice_name_request = GetSpiceNameRequest()
            spice_name_service_handle = rospy.ServiceProxy('get_spice_name_service', GetSpiceName)
            print("[spiceUpCoordinator] : Requesting target spice name")
            spice_name_service_response = spice_name_service_handle(spice_name_request)
            target_spice = spice_name_service_response.target_spice_name
            print("[spiceUpCoordinator] : Received target spice name: "+str(target_spice))

            target_location_index = self.spice_index_mapping.get(target_spice)
            print("[spiceUpCoordinator] : Target spice index: "+str(target_location_index))
            
            # Fill result
            print(self.poseProcessor.grasp_msg_dict) # TODO this array is filled with Falses
            grasp_msg = self.poseProcessor.grasp_msg_dict[target_location_index]
            print(grasp_msg)
            result.ee_pickup_target = grasp_msg
            result.ee_intermediary_target = self.intermediary_target
            result.ee_dropoff_target = self.poseProcessor.drop_off_msg_dict[self.drop_off_index]
            result.ee_pre_target = self.poseProcessor.approach_msg_dict[target_location_index]

            # stamp over all msgs
            result.ee_pickup_target.header.stamp = self.stamp
            result.ee_intermediary_target.header.stamp = self.stamp
            result.ee_dropoff_target.header.stamp = self.stamp
            result.ee_pre_target.header.stamp = self.stamp

            # match the orientation of the dropoff to that of the pickup
            result.ee_dropoff_target.pose.orientation = result.ee_pickup_target.pose.orientation

            # Broadcast grasp pose to tf-tree
            # br = tf.TransformBroadcaster()

            br.sendTransform((grasp_msg.pose.position.x, 
                    grasp_msg.pose.position.y, 
                    grasp_msg.pose.position.z),
                    (grasp_msg.pose.orientation.x,
                    grasp_msg.pose.orientation.y,
                    grasp_msg.pose.orientation.z,
                    grasp_msg.pose.orientation.w),
                    rospy.Time.now(),
                    "spice_up_grasp",
                    grasp_msg.header.frame_id)

            br.sendTransform((self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.position.x, 
                    self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.position.y, 
                    self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.position.z),
                    (self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.orientation.x,
                    self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.orientation.y,
                    self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.orientation.z,
                    self.poseProcessor.drop_off_msg_dict[self.drop_off_index].pose.orientation.w),
                    rospy.Time.now(),
                    "spice_up_dropoff",
                    grasp_msg.header.frame_id)
            
            # Set action status to succeeded and transfer result to action server
            self.action_server.set_succeeded(result)
            
            # Visualization
            pose_viz_img = self.poseProcessor.get_specific_viz(target_location_index,self.drop_off_index)
            if pose_viz_img is not None:
                pose_visualized_msg = self.cv2_to_ros(pose_viz_img)
                self.pose_debug_pub.publish(pose_visualized_msg)
                # debug_imgs_path = rospy.get_param("index_finder/HOME") + "debug_imgs/pose"+str(self.drop_off_index)+"_viz.png"
                # cv2.imwrite(debug_imgs_path,pose_viz_img)

            self.drop_off_index += 1 # Use the second drop off pose for the next request

            # Shutdown
            if self.drop_off_index == 2:
                self.shutdown("Job done")

    def load_params(self):
        
        self.debug  = rospy.get_param("spice_up_coordinator/debug")

        self.poseProcessor = None 

        self.drop_off_index = 0 # Index of drop of pose. After first drop off pose is computed, this variable is incremented by one. Once it reaches two, the node is killed
        self.spice_index_mapping = {
            "pepper": 0,
            "salt": 1,
            "oil": 2,
            "vinegar": 3
        }

        self.mask_msg = None

        self._bridge = CvBridge()

        self.last_image_color_msg = None
        self.last_image_depth = None
        self.last_image_depth_msg = None

        sim = rospy.get_param("spice_up_coordinator/in_simulation_mode")
        if sim:
            self.camera_info_topic_name = "/camera/color/camera_info"
            self.color_topic_name = "/camera/color/image_raw"
            self.depth_topic_name = "/camera/aligned_depth_to_color/image_raw"
        else:
            self.camera_info_topic_name = "/dynaarm_REALSENSE/color/camera_info"
            self.color_topic_name = "/dynaarm_REALSENSE/color/image_raw"
            self.depth_topic_name = "/dynaarm_REALSENSE/aligned_depth_to_color/image_raw"

        self.K = self.get_intrinsics()

        # hardcoded intermediary point so that we don't collide with the shelf 0.702, 0.004, 0.712, 0.005
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base"
        pose_msg.pose.position.x = 0.63
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.499
        pose_msg.pose.orientation.x = 0.702
        pose_msg.pose.orientation.y = 0.004
        pose_msg.pose.orientation.z = 0.712
        pose_msg.pose.orientation.w = 0.005
        self.intermediary_target = pose_msg

    # Utils -----------------------------------------------------------    

    def get_intrinsics(self):
        try:
            data = rospy.wait_for_message(self.camera_info_topic_name, CameraInfo, timeout=10.0)
            K = np.array(data.K).reshape(3, 3).astype(np.float64)
            return K
        except rospy.ROSException:
            rospy.logwarn(f"[SpiceUpActionServer]: Failed to get intrinsics from topic '{self.camera_info_topic_name}', retrying...")
            return self.get_intrinsics()
    
    def synch_image_callback(self, color_msg, depth_msg):
        try:
            cv_depth_img = self._bridge.imgmsg_to_cv2(depth_msg,desired_encoding = "passthrough").astype(np.uint16).copy() / 1000.0
            cv_depth_img[(cv_depth_img < 0.1)] = 0
            self.last_image_depth = cv_depth_img
            self.last_image_depth_msg = depth_msg

        except CvBridgeError as e:
            print("[spiceUpCoordinator] : Error on the depth image decoding! : "+str(e))
        
        try:
            cv_color_img = self._bridge.imgmsg_to_cv2(color_msg,desired_encoding="bgr8")
            self.last_image_color = cv_color_img
            self.last_image_color_msg = color_msg

        except CvBridgeError as e:
            print("[spiceUpCoordinator] : Error on the color image decoding! : "+str(e))

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    def shutdown_cb(self, msg):
        print("[spiceUpCoordinator] : External Shutdown requested!")
        print("[spiceUpCoordinator] : Shutting down")
        result = SpiceUpBottlePickResult()
        self.action_server.set_aborted(result)
        rospy.signal_shutdown("External Shutdown requested")
    
    def shutdown(self,msg):
        print("[spiceUpCoordinator] : Sending shutdown request")
        shutdown_msg = Bool()
        shutdown_msg.data = True
        self.shutdown_pub.publish(shutdown_msg)

        print("[spiceUpCoordinator] : Shutting down")
        rospy.signal_shutdown(msg)
    # -----------------------------------------------------------------

if __name__ == '__main__':
    rospy.init_node('spice_up_action_server')
    server = spiceUpCoordinator()
    rospy.spin()
