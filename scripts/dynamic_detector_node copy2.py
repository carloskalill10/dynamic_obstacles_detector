#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import least_squares, linear_sum_assignment
from sklearn.cluster import DBSCAN
from tf.transformations import euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point

class Track:
    def __init__(self, track_id, x, y):
        self.track_id = track_id
        # Estado do KF: [x, y, vx, vy]
        self.state = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 1.0  # Matriz de Covariancia inicial
        self.missed_frames = 0    # Contador para deletar objetos perdidos
        
    def predict(self, dt):
        # Matriz de transicao de estado 
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # Ruido do processo de estimativa
        Q = np.eye(4) * 0.1 
        
        self.state = F.dot(self.state)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, measurement):
        # Matriz de observacao (So mede X e Y, nao velocidade)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # Ruido da medicao (Confianca no DBSCAN)
        R = np.eye(2) * 0.1 
        
        Z = np.array(measurement)
        Y = Z - H.dot(self.state)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        self.state = self.state + K.dot(Y)
        self.P = (np.eye(4) - K.dot(H)).dot(self.P)
        self.missed_frames = 0 # Zera o contador 

        max_vel = 1.5 # Limite em m/s para os obstaculos 
        speed = np.hypot(self.state[2], self.state[3])
        if speed > max_vel:
            self.state[2] = (self.state[2] / speed) * max_vel
            self.state[3] = (self.state[3] / speed) * max_vel



class LidarCircleFittingDetector:
    def __init__(self):
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('dynamic_detector', anonymous=True)

        # Buffer para comparacao temporal (T a T-5)
        self.history_buffer = []
        self.buffer_size = 10
        
        # Parametros
        self.threshold = 0.12
        self.max_dist_detect = 6
        self.max_limit_dist = 1.0
        self.raio_do_robo = 0.35 #como a lidar fica dentro do raio do robo, deve-se considerar como area minima de deteccao o raio do robo.

        #variaveis do tracking
        self.tracks = []
        self.next_track_id = 0
        self.max_missed_frames = 5   # Tolerancia antes de deletar o ID
        self.max_assoc_dist = 0.4    # Distancia maxima (metros) para associar um ponto ao Track
        self.last_time = None        # Para calcular o 'dt' do Filtro de Kalman

        # Buffer de TF2 (No ROS1, o listener precisa ser mantido vivo na classe)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher para o RViz
        self.marker_pub = rospy.Publisher('detection_markers', MarkerArray, queue_size=10)
        
        # Subscriber do Lidar
        self.scan_sub = rospy.Subscriber('/front/scan', LaserScan, self.scan_callback, queue_size=1, buff_size=2**24)

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

        current_time = msg.header.stamp

        # 1. Obter a pose sincronizada via TF2
        try:
            # lookup_transform usa rospy.Time(0) para o mais recente ou msg.header.stamp
            transform = self.tf_buffer.lookup_transform(
                'odom', 
                msg.header.frame_id, 
                msg.header.stamp, #rospy.Time(0),
                rospy.Duration(0.1)# deixei 0 devido que o filtro ja garante a recepcao dos dados

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

        # Inicializa o tempo na primeira iteracao
        if self.last_time is None:
            self.last_time = current_time

        # Calcula o delta time (dt)
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time


        # Inicializacao das posicoes do buffer
        if len(self.history_buffer) < self.buffer_size:
            self.history_buffer.append(current_frame_data)
            return
        
        prev_data = self.history_buffer[0]
        prev_ranges = prev_data['ranges']
        prev_pose = prev_data['pose']
        prev_angles = prev_data['angles']

        # 3. Processamento espacial
        mask_prev = (prev_ranges > self.raio_do_robo) & (prev_ranges < msg.range_max)
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
        mask_curr = (curr_ranges > self.raio_do_robo) & (curr_ranges < msg.range_max)
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

        current_detections = []
        if len(dynamic_points_coords) >= 7:
            clustering = DBSCAN(eps=0.4, min_samples=4).fit(dynamic_points_coords)
            for label in set(clustering.labels_):
                if label == -1: continue
                obj_pts = dynamic_points_coords[clustering.labels_ == label]
                xc, yc, _ = self.fit_circle_least_squares(obj_pts)
                current_detections.append((xc, yc))
        
       
       
        # 6. RASTREAMENTO (TRACKING)
        # 6.1 Predicao do Filtro de Kalman
        if dt > 0:
            for track in self.tracks:
                track.predict(dt)

        # 6.2 Associacao de Dados (Algoritmo Hungaro)
        unmatched_detections = []
        if len(self.tracks) == 0:
            unmatched_detections = current_detections
        elif len(current_detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(current_detections)))
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(current_detections):
                    dist = np.hypot(track.state[0] - det[0], track.state[1] - det[1])
                    cost_matrix[t, d] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for t, d in zip(row_ind, col_ind):
                if cost_matrix[t, d] < self.max_assoc_dist:
                    self.tracks[t].update(current_detections[d])
                    matched_tracks.add(t)
                    matched_dets.add(d)

            unmatched_detections = [current_detections[i] for i in range(len(current_detections)) if i not in matched_dets]

            for t in range(len(self.tracks)):
                if t not in matched_tracks:
                    self.tracks[t].missed_frames += 1
        else:
            for track in self.tracks:
                track.missed_frames += 1

        # 6.3 Criar novos tracks
        for det in unmatched_detections:
            new_track = Track(self.next_track_id, det[0], det[1])
            self.tracks.append(new_track)
            self.next_track_id += 1

        # 6.4 Deletar tracks perdidos
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed_frames]

        
        # 7. PUBLICAR MARCADORES NO RVIZ
        # ----------------------------------------------------
        marker_array = MarkerArray()
        
        # Limpar marcadores antigos
        clear_msg = Marker()
        clear_msg.action = Marker.DELETEALL
        clear_msg.header.frame_id = "odom"  # Simplificado para odom generico
        marker_array.markers.append(clear_msg)

        marker_array.markers.append(self.create_robot_marker(curr_pose))

        # Criar marcadores para os obstaculos rastreados
        for track in self.tracks:
            xc_est, yc_est = track.state[0], track.state[1]
            vx_est, vy_est = track.state[2], track.state[3]
            speed = np.hypot(vx_est, vy_est)
            
            # Cilindro base
            marker_array.markers.append(self.create_cylinder_marker(xc_est, yc_est, 0.5, track.track_id))
            # Texto do ID
            marker_array.markers.append(self.create_cylinder_marker(xc_est, yc_est, 0.5, track.track_id + 1000, is_text=True))

            # Seta de Velocidade (So desenha se estiver se movendo)
            if speed > 0.05:
                # Multiplicamos a velocidade por um fator (ex: 1.0) para alongar a seta visualmente se necessario
                marker_array.markers.append(self.create_velocity_arrow_marker(xc_est, yc_est, vx_est * 1.0, vy_est * 1.0, track.track_id + 2000))

            rospy.loginfo("Obstaculo ID {0} | Pos: ({1:.2f}, {2:.2f}) | Vel: {3:.2f} m/s".format(
                track.track_id, xc_est, yc_est, speed))

        if len(self.tracks) > 0:
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
            marker.scale.x = 2*r
            marker.scale.y = 2*r
            marker.scale.z = r
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8
            
        marker.lifetime = rospy.Duration(0.2)
        return marker
    

    def create_velocity_arrow_marker(self, x, y, vx, vy, obj_id):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "velocities"
        marker.id = obj_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Inicio da seta (centro do obstaculo)
        p_start = Point()
        p_start.x = x
        p_start.y = y
        p_start.z = 0.5 # Mesma altura do centro do cilindro
        
        # Fim da seta (posicao atual + vetor velocidade)
        p_end = Point()
        p_end.x = x + vx
        p_end.y = y + vy
        p_end.z = 0.5
        
        marker.points.append(p_start)
        marker.points.append(p_end)
        
        # Escala da seta (haste, ponta, comprimento da ponta)
        marker.scale.x = 0.1
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        # Cor amarela
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(0.2)
        return marker

if __name__ == '__main__':
    try:
        detector = LidarCircleFittingDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass