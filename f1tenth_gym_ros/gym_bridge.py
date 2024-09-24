# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
import rclpy
from rclpy.node import Node

import rclpy.publisher
import rclpy.subscription
import rclpy.timer
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Float64MultiArray

import f1tenth_gym
import gymnasium as gym
import numpy as np
from transforms3d import euler

class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')

        self.declare_parameter('ego_namespace', 'ego_racecar')
        self.declare_parameter('drive_topic', 'drive')
        self.declare_parameter('state_topic', 'state')
        self.declare_parameter('sx', 0.0)
        self.declare_parameter('sy', 0.0)
        self.declare_parameter('stheta', 0.0)
    
        self.declare_parameter('opp_namespace', 'opp_racecar')
        self.declare_parameter('sx1', 2.0)
        self.declare_parameter('sy1', 0.5)
        self.declare_parameter('stheta1', 0.0)

        self.declare_parameter('map_path', '/home/michal/Documents/f1tenth_gym_ros/f1tenth_gym_ros/maps/Spielberg_map')
        self.declare_parameter('map_img_ext', '.png')
        

        # env backend
        self.env: gym.Env = gym.make('f1tenth-v0',
                            map=self.get_parameter('map_path').value,
                            map_ext=self.get_parameter('map_img_ext').value,
                            num_agents=2,
                            config={
                                    "control_input": ["speed", "steering_angle"],
                                    "observation_config": {"type": "dynamic_state"},
                                    },)
        
        drive_topic: str = self.get_parameter('drive_topic').value
        state_topic: str = self.get_parameter('state_topic').value

        self.ego_namespace: str = self.get_parameter('ego_namespace').value
        sx: float = self.get_parameter('sx').value
        sy: float = self.get_parameter('sy').value
        stheta: float = self.get_parameter('stheta').value
        self.ego_pose: list[float] = [sx, sy, stheta]
        self.ego_requested_speed: float = 0.0
        self.ego_steer: float = 0.0
        self.ego_collision: bool = False
        self.ego_drive_published: bool = False

        self.opp_namespace: str = self.get_parameter('opp_namespace').value
        sx1: float = self.get_parameter('sx1').value
        sy1: float = self.get_parameter('sy1').value
        stheta1: float = self.get_parameter('stheta1').value
        self.opp_pose: list[float] = [sx1, sy1, stheta1]
        self.opp_requested_speed: float = 0.0
        self.opp_steer: float = 0.0
        self.opp_collision: bool = False

        self.obs, self.info = self.env.reset(options={'poses' : np.array([[sx, sy, stheta], [sx1, sy1, stheta1]])})
        self._update_sim_state()

        # sim physical step timer
        self.drive_timer: rclpy.timer.Timer = self.create_timer(0.01, self.drive_timer_callback)
        # topic publishing timer
        self.timer: rclpy.timer.Timer = self.create_timer(0.004, self.timer_callback)

        # transform broadcaster
        self.br: TransformBroadcaster = TransformBroadcaster(self)

        self.ego_state_publisher: rclpy.publisher.Publisher = self.create_publisher(Float64MultiArray, f'{self.ego_namespace}/{state_topic}', 10)
        self.opp_state_publisher: rclpy.publisher.Publisher = self.create_publisher(Float64MultiArray, f'{self.opp_namespace}/{state_topic}', 10)

        # subscribers
        self.ego_drive_sub: rclpy.subscription.Subscription = self.create_subscription(AckermannDriveStamped, f'{self.ego_namespace}/{drive_topic}', self.drive_callback, 10)
        self.ego_reset_sub: rclpy.subscription.Subscription = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.ego_reset_callback, 10)

        self.opp_drive_sub: rclpy.subscription.Subscription = self.create_subscription(AckermannDriveStamped, f'{self.opp_namespace}/{drive_topic}', self.opp_drive_callback, 10)
        self.opp_reset_sub: rclpy.subscription.Subscription = self.create_subscription(PoseStamped, '/goal_pose', self.opp_reset_callback, 10)


    def drive_callback(self, drive_msg: AckermannDriveStamped):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True


    def opp_drive_callback(self, drive_msg: AckermannDriveStamped):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True


    def ego_reset_callback(self, pose_msg: PoseWithCovarianceStamped):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')

        opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
        self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta], opp_pose]))


    def opp_reset_callback(self, pose_msg: PoseWithCovarianceStamped):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        self.obs, _ , self.done, _ = self.env.reset(np.array([list(self.ego_pose), [rx, ry, rtheta]]))


    def drive_timer_callback(self):
        self.obs, _, self.done, _, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        self._update_sim_state()


    def timer_callback(self):
        ts = self.get_clock().now().to_msg()
        self._publish_transforms(ts)
        self._publish_wheel_transforms(ts)
        self._publish_states()

    def _update_sim_state(self):
        self.opp_pose[0] = self.obs['agent_0']['pose_x']
        self.opp_pose[1] = self.obs['agent_0']['pose_y']
        self.opp_pose[2] = self.obs['agent_0']['pose_theta']

        self.ego_pose[0] = self.obs['agent_1']['pose_x']
        self.ego_pose[1] = self.obs['agent_1']['pose_y']
        self.ego_pose[2] = self.obs['agent_1']['pose_theta']

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = float(self.ego_pose[0])
        ego_t.translation.y = float(self.ego_pose[1])
        ego_t.translation.z = float(0.0)
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = float(ego_quat[1])
        ego_t.rotation.y = float(ego_quat[2])
        ego_t.rotation.z = float(ego_quat[3])
        ego_t.rotation.w = float(ego_quat[0])

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        opp_t = Transform()
        opp_t.translation.x = float(self.opp_pose[0])
        opp_t.translation.y = float(self.opp_pose[1])
        opp_t.translation.z = float(0.0)
        opp_quat = euler.euler2quat(0.0, 0.0, self.opp_pose[2], axes='sxyz')
        opp_t.rotation.x = float(opp_quat[1])
        opp_t.rotation.y = float(opp_quat[2])
        opp_t.rotation.z = float(opp_quat[3])
        opp_t.rotation.w = float(opp_quat[0])

        opp_ts = TransformStamped()
        opp_ts.transform = opp_t
        opp_ts.header.stamp = ts
        opp_ts.header.frame_id = 'map'
        opp_ts.child_frame_id = self.opp_namespace + '/base_link'
        self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = float(ego_wheel_quat[1])
        ego_wheel_ts.transform.rotation.y = float(ego_wheel_quat[2])
        ego_wheel_ts.transform.rotation.z = float(ego_wheel_quat[3])
        ego_wheel_ts.transform.rotation.w = float(ego_wheel_quat[0])
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        opp_wheel_ts = TransformStamped()
        opp_wheel_quat = euler.euler2quat(0., 0., self.opp_steer, axes='sxyz')
        opp_wheel_ts.transform.rotation.x = float(opp_wheel_quat[1])
        opp_wheel_ts.transform.rotation.y = float(opp_wheel_quat[2])
        opp_wheel_ts.transform.rotation.z = float(opp_wheel_quat[3])
        opp_wheel_ts.transform.rotation.w = float(opp_wheel_quat[0])
        opp_wheel_ts.header.stamp = ts
        opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
        opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
        self.br.sendTransform(opp_wheel_ts)
        opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
        opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
        self.br.sendTransform(opp_wheel_ts)

    def _publish_states(self):
        ego = Float64MultiArray()
        ego.data = [self.obs['agent_0']['pose_x'],
                    self.obs['agent_0']['pose_y'],
                    self.obs['agent_0']['delta'],
                    self.obs['agent_0']['linear_vel_x'],
                    self.obs['agent_0']['pose_theta'],
                    self.obs['agent_0']['ang_vel_z'],
                    self.obs['agent_0']['beta']]
        self.ego_state_publisher.publish(ego)

        opp = Float64MultiArray()
        opp.data = [self.obs['agent_1']['pose_x'],
                    self.obs['agent_1']['pose_y'],
                    self.obs['agent_1']['delta'],
                    self.obs['agent_1']['linear_vel_x'],
                    self.obs['agent_1']['pose_theta'],
                    self.obs['agent_1']['ang_vel_z'],
                    self.obs['agent_1']['beta']]
        self.opp_state_publisher.publish(opp)

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()
