#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from tf.transformations import euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs

class LidarCircleFittingDetector:
    def __init__(self):
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('dynamic_detector', anonymous=True)

        # Buffer para comparacao temporal (T a T-5)
        self.history_buffer = []
        self.buffer_size = 5 
        
        # Parametros
        self.threshold = 0.15 
        self.max_dist_detect = 3
        self.max_limit_dist = 1.0

        # Buffer de TF2 (No ROS1, o listener precisa ser mantido vivo na classe)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher para o RViz
        self.marker_pub = rospy.Publisher('detection_markers', MarkerArray, queue_size=10)
        
        # Subscriber do Lidar
        self.scan_sub = rospy.Subscriber('/front/scan', LaserScan, self.scan_callback)

        rospy.loginfo('No de deteccao dinamica foi iniciado...')

    def fit_circle_least_squares(self, points):
        def calc_dist(params, x, y):
            xc, yc, R = params
            return np.sqrt((x - xc)**2 + (y - yc)**2) - R
        x, y = points[:, 0], points[:, 1]
        initial_guess = [np.mean(x), np.mean(y), 0.1] 
        result = least_squares(calc_dist, initial_guess, args=(x, y))
        return result.x

    def scan_callback(self, msg):
        # 1. Obter a pose sincronizada via TF2
        try:
            # lookup_transform usa rospy.Time(0) para o mais recente ou msg.header.stamp
            transform = self.tf_buffer.lookup_transform(
                'odom', 
                msg.header.frame_id, 
                msg.header.stamp,#rospy.Time(0), 
                rospy.Duration(0.2)
            )
            
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, tyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            curr_pose = [tx, ty, tyaw]
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn('Aguardando transformacao TF: %s', e)
            return

        current_frame_data = {
            'ranges': np.array(msg.ranges),
            'pose': curr_pose,
            'angles': np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        }

        if len(self.history_buffer) < self.buffer_size:
            self.history_buffer.append(current_frame_data)
            return
        
        prev_data = self.history_buffer[0]
        prev_ranges = prev_data['ranges']
        prev_pose = prev_data['pose']
        prev_angles = prev_data['angles']

        # 3. Processamento espacial
        mask_prev = (prev_ranges > msg.range_min) & (prev_ranges < msg.range_max)
        if not np.any(mask_prev):
            self.update_buffer(current_frame_data)
            return

        px_re = prev_ranges[mask_prev] * np.cos(prev_angles[mask_prev])
        py_re = prev_ranges[mask_prev] * np.sin(prev_angles[mask_prev])

        p_world_x = prev_pose[0] + (px_re * np.cos(prev_pose[2]) - py_re * np.sin(prev_pose[2]))
        p_world_y = prev_pose[1] + (px_re * np.sin(prev_pose[2]) + py_re * np.cos(prev_pose[2]))
        
        dx = p_world_x - curr_pose[0]
        dy = p_world_y - curr_pose[1]
        x_local_past = dx * np.cos(-curr_pose[2]) - dy * np.sin(-curr_pose[2])
        y_local_past = dx * np.sin(-curr_pose[2]) + dy * np.cos(-curr_pose[2])

        # 4. Correlacao
        r_past = np.sqrt(x_local_past**2 + y_local_past**2)
        theta_past = np.arctan2(y_local_past, x_local_past)
        
        sort_idx = np.argsort(theta_past)
        theta_past_sorted = theta_past[sort_idx]
        r_past_sorted = r_past[sort_idx]

        curr_ranges = current_frame_data['ranges']
        curr_angles = current_frame_data['angles']
        mask_curr = (curr_ranges > msg.range_min) & (curr_ranges < msg.range_max)
        curr_angles_valid = curr_angles[mask_curr]

        if len(theta_past_sorted) == 0 or len(curr_angles_valid) == 0:
            self.update_buffer(current_frame_data)
            return

        r_predito = np.interp(curr_angles_valid, theta_past_sorted, r_past_sorted)
        dists_diff = np.abs(curr_ranges[mask_curr] - r_predito)
        dynamic_mask = (dists_diff > self.threshold) & (dists_diff < self.max_limit_dist) & (curr_ranges[mask_curr] < self.max_dist_detect)

        # 5. Agrupamento e Marcadores
        curr_x_local = curr_ranges[mask_curr] * np.cos(curr_angles_valid)
        curr_y_local = curr_ranges[mask_curr] * np.sin(curr_angles_valid)
        dyn_x_local = curr_x_local[dynamic_mask]
        dyn_y_local = curr_y_local[dynamic_mask]
        
        p_dyn_world_x = curr_pose[0] + (dyn_x_local * np.cos(curr_pose[2]) - dyn_y_local * np.sin(curr_pose[2]))
        p_dyn_world_y = curr_pose[1] + (dyn_x_local * np.sin(curr_pose[2]) + dyn_y_local * np.cos(curr_pose[2]))
        dynamic_points_coords = np.column_stack((p_dyn_world_x, p_dyn_world_y))

        marker_array = MarkerArray()
        clear_msg = Marker()
        clear_msg.action = Marker.DELETEALL
        clear_msg.header.frame_id = "/jackal_velocity_controller/odom"
        marker_array.markers.append(clear_msg)

        marker_array.markers.append(self.create_robot_marker(curr_pose))

        if len(dynamic_points_coords) >= 7:
            clustering = DBSCAN(eps=0.3, min_samples=7).fit(dynamic_points_coords)
            for label in set(clustering.labels_):
                if label == -1: continue
                obj_pts = dynamic_points_coords[clustering.labels_ == label]
                xc, yc, _ = self.fit_circle_least_squares(obj_pts)

            
                marker_array.markers.append(self.create_cylinder_marker(xc, yc, 0.5, int(label)))
                marker_array.markers.append(self.create_cylinder_marker(xc, yc, 0.5, int(label) + 100, is_text=True))

                rospy.loginfo("Obstaculo {0} detectado: x={1:.2f}, y={2:.2f} ".format(label, xc, yc))

            rospy.loginfo("------------")
        self.marker_pub.publish(marker_array)
        self.update_buffer(current_frame_data)

    def update_buffer(self, frame_data):
        self.history_buffer.append(frame_data)
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)

    def create_robot_marker(self, pose):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot"
        marker.id = 200
        marker.type = Marker.ARROW
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.orientation.z = np.sin(pose[2] / 2.0)
        marker.pose.orientation.w = np.cos(pose[2] / 2.0)
        marker.scale.x = 0.5; marker.scale.y = 0.1; marker.scale.z = 0.1
        marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0; marker.color.a = 1.0
        return marker

    def create_cylinder_marker(self, xc, yc, r, obj_id, is_text=False):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacles"
        marker.id = obj_id
        marker.pose.position.x = xc
        marker.pose.position.y = yc
        
        if is_text:
            marker.type = Marker.TEXT_VIEW_FACING
            marker.text = "Dinamico {}".format(obj_id % 100)
            marker.pose.position.z = 1.0
            marker.scale.z = 0.2
            marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0
        else:
            marker.type = Marker.CYLINDER
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8
            
        marker.lifetime = rospy.Duration(0.2)
        return marker

if __name__ == '__main__':
    try:
        detector = LidarCircleFittingDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass