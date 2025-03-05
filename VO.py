import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm


class VisualOdometry:
    def __init__(self, data_dir, real_time):

        self.real_time = real_time
        self.data_dir = data_dir

        self.K_l, self.P_l, self.K_r, self.P_r = None, None, None, None
        self.loadCalibration(data_dir + '/sequences/01/calib.txt')

        self.baseline = 0.54  # abs(P_l[0, 3] - P_r[0, 3]) / K_l[0, 0]

        self.ground_truth = self.loadGroundTruth(os.path.join(data_dir, 'poses/01.txt'))
        self.distanceReal = self.calculateDistanceGT(100)

        # FAST Features from Accelerated Segment Test
        self.fast = cv2.FastFeatureDetector_create()

        # Lucas-Kanade parameters used for tracking keypoints between frames
        self.lk_params = dict(winSize=(7, 7),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                              )

        self.disparity = []

        if not real_time:
            self.image_dir_l = os.path.join(data_dir, 'sequences/01/image_0')
            self.image_dir_r = os.path.join(data_dir, 'sequences/01/image_1')
            self.num_images = len(os.listdir(self.image_dir_l))
        else:
            self.ground_truth = None  # No ground truth in real-time mode

        # To accumulate trajectory; initialize with identity
        self.trajectory = [np.eye(4, dtype=np.float64)]
        # Cumulative distance traveled (in meters)
        self.cumulative_distance = 0.0

    def handle_transformation(self, R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def loadCalibration(self, calib_path):
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if line.startswith("P0:"):
                    self.P_l = np.array(line.strip().split()[1:], dtype=np.float32).reshape(3, 4)
                elif line.startswith("P1:"):
                    self.P_r = np.array(line.strip().split()[1:], dtype=np.float32).reshape(3, 4)
        self.K_l = self.P_l[0:3, 0:3]
        self.K_r = self.P_r[0:3, 0:3]

    def loadGroundTruth(self, ground_truth_path):
        poses = []
        with open(ground_truth_path, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    def calculateDistanceGT(self, number_of_frames):
        total_distance = 0.0
        for i in tqdm(range(1, min(len(self.ground_truth), number_of_frames))):
            previous_pose = self.ground_truth[i - 1][:3, 3]
            current_pose = self.ground_truth[i][:3, 3]
            distance = np.linalg.norm(current_pose - previous_pose)
            total_distance += distance
        return total_distance

    def loadImages(self, filepath):
        image_paths = [os.path.join(filepath, file) for file in tqdm(sorted(os.listdir(filepath)), desc=filepath)]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in tqdm(image_paths)]
        return images

    def loadImage(self, image_dir, idx):
        """Loads a single image given the index."""
        img_path = os.path.join(image_dir, f"{idx:06d}.png")
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def showImage(self, idx, img_d):
        if idx < len(img_d):
            img = self.images_l[idx]
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Index out of range.")

    def debugDisparity(self, frame_idx=100):
        img_l = self.loadImage(self.image_dir_l, frame_idx)
        img_r = self.loadImage(self.image_dir_r, frame_idx)
        disp = self.compute_disparity_and_depth(img_l, img_r)[0]
        plt.figure(figsize=(10, 5))
        plt.imshow(disp, cmap='jet')
        plt.colorbar()
        plt.title(f"Disparity Map - Frame {frame_idx}")
        plt.show()

    def plotTrajectory(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        if self.ground_truth:
            gt_x = [pose[0, 3] for pose in self.ground_truth]
            gt_z = [pose[2, 3] for pose in self.ground_truth]
            ax.plot(gt_x, gt_z, label="Ground Truth", color="blue", linewidth=2)
        traj_x = [T[0, 3] for T in self.trajectory]
        traj_z = [T[2, 3] for T in self.trajectory]
        ax.plot(traj_x, traj_z, label="Estimated Trajectory", color="red", linewidth=2)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Z (meters)")
        ax.set_title("Camera Trajectory")
        ax.legend()
        ax.grid()
        plt.show()

    def compute_disparity_and_depth(self, img_l, img_r):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=15, P1=8 * 15 ** 2, P2=32 * 15 ** 2,
            disp12MaxDiff=10, uniquenessRatio=5, speckleWindowSize=200, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = np.divide(stereo.compute(img_l, img_r).astype(np.float32), 16)
        disparity[disparity < 1] = 1e-6
        focal_length = self.K_l[0, 0]
        baseline = self.baseline
        depth = np.divide(focal_length * baseline, disparity,
                          out=np.zeros_like(disparity), where=disparity > 0)
        return disparity, depth

    def update_trajectory(self, transformation_matrix):
        # Update the cumulative trajectory and distance.
        new_pose = self.trajectory[-1] @ transformation_matrix
        self.trajectory.append(new_pose)
        # Calculate incremental distance from translation vector.
        distance = np.linalg.norm(transformation_matrix[:3, 3])
        self.cumulative_distance += distance
        return new_pose

    def divide_into_tiles(self, img, tile_h, tile_w):
        tiles = []
        tile_positions = []
        h, w = img.shape
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                tile = img[y:y + tile_h, x:x + tile_w]
                tiles.append(tile)
                tile_positions.append((x, y))
        return tiles, tile_positions

    def extract_keypoints(self, image, tile_h, tile_w):
        tiles, tile_positions = self.divide_into_tiles(image, tile_h, tile_w)
        keypoints = []
        for (tile, (x_offset, y_offset)) in zip(tiles, tile_positions):
            kp = self.fast.detect(tile, None)
            for pt in kp:
                adjusted_pt = cv2.KeyPoint(pt.pt[0] + x_offset, pt.pt[1] + y_offset,
                                           pt.size, pt.angle, pt.response, pt.octave, pt.class_id)
                keypoints.append(adjusted_pt)
        return keypoints

    def track_keypoints_lk(self, image1, image2, keypoints, max_error=4):
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(keypoints), 1)
        trackpoints2, status, error = cv2.calcOpticalFlowPyrLK(image1, image2, trackpoints1, None, **self.lk_params)
        trackable = status.astype(bool)
        under_threshhold = np.where(error[trackable] < max_error, True, False)
        trackpoints1 = trackpoints1[trackable][under_threshhold]
        trackpoints2 = np.around(trackpoints2[trackable][under_threshhold])
        h, w = image1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]
        return trackpoints1, trackpoints2

    def get_valid_disparities(self, q, disp, min_disp=0.0, max_disp=100.0):
        q_idx = q.astype(int)
        disp_values = disp.T[q_idx[:, 0], q_idx[:, 1]]
        valid_mask = np.logical_and(min_disp < disp_values, disp_values < max_disp)
        return disp_values, valid_mask

    def compute_right_keypoints(self, q_left, disp_values):
        q_right = np.copy(q_left)
        q_right[:, 0] -= disp_values
        return q_right

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        disp1_values, mask1 = self.get_valid_disparities(q1, disp1, min_disp, max_disp)
        disp2_values, mask2 = self.get_valid_disparities(q2, disp2, min_disp, max_disp)
        in_bounds = np.logical_and(mask1, mask2)
        q1_l = q1[in_bounds]
        q2_l = q2[in_bounds]
        disp1_valid = disp1_values[in_bounds]
        disp2_valid = disp2_values[in_bounds]
        q1_r = self.compute_right_keypoints(q1_l, disp1_valid)
        q2_r = self.compute_right_keypoints(q2_l, disp2_valid)
        return q1_l, q1_r, q2_l, q2_r

    def triangulate_points(self, q1_l, q2_l, q1_r, q2_r):
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        transf = self.handle_transformation(R, t)
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])
        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]
        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        early_termination_threshold = 5
        min_error = float('inf')
        early_termination = 0
        out_pose = None

        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6, replace=False)
            sample_q1 = q1[sample_idx]
            sample_q2 = q2[sample_idx]
            sample_Q1 = Q1[sample_idx]
            sample_Q2 = Q2[sample_idx]
            initial_guess = np.zeros(6)
            opt_res = least_squares(self.reprojection_residuals, initial_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))
            residuals = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            residuals = residuals.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(residuals, axis=1))
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination >= early_termination_threshold:
                break

        r = out_pose[:3]
        t = out_pose[3:]
        R, _ = cv2.Rodrigues(r)
        transformation_matrix = self.handle_transformation(R, t)
        return transformation_matrix

    def get_pose(self, index):
        image_l1 = self.loadImage(self.image_dir_l, index)
        image_r1 = self.loadImage(self.image_dir_r, index)
        image_l2 = self.loadImage(self.image_dir_l, index + 1)
        image_r2 = self.loadImage(self.image_dir_r, index + 1)

        disparity1, depth1 = self.compute_disparity_and_depth(image_l1, image_l2)
        disparity2, depth2 = self.compute_disparity_and_depth(image_r1, image_r2)

        keypoint1_l = self.extract_keypoints(image_l1, 10, 20)
        trackpoints1, trackpoints2 = self.track_keypoints_lk(image_l1, image_l2, keypoint1_l)

        q1_l, q1_r, q2_l, q2_r = self.calculate_right_qs(trackpoints1, trackpoints2,
                                                           disparity1, disparity2)
        Q1, Q2 = self.triangulate_points(q1_l, q2_l, q1_r, q2_r)
        pose = self.estimate_pose(q1_l, q2_l, Q1, Q2, max_iter=100)

        # Update the cumulative trajectory using the new pose transformation.
        new_pose = self.update_trajectory(pose)
        # Calculate the incremental distance traveled for this frame.

        return new_pose

    def estimate_path(self, start_idx=0, end_idx=None):
        """
        Estimates the camera path for a range of image indices.

        Parameters
        ----------
        start_idx : int
            The starting frame index.
        end_idx : int, optional
            The ending frame index. If not provided, defaults to the last image.

        Returns
        -------
        path : list of tuple
            A list of (x, z) positions representing the estimated trajectory.
        """
        if end_idx is None:
            end_idx = self.num_images - 1

        self.trajectory = [np.eye(4, dtype=np.float64)]  # Start with identity matrix

        path = []  # Store (x, z) positions

        for idx in tqdm(range(start_idx, end_idx)):
            new_pose = self.get_pose(idx)
            self.trajectory.append(new_pose)  # Store the transformation matrix

            # Extract position (x, z) from the transformation matrix
            x, z = new_pose[0, 3], new_pose[2, 3]
            path.append((x, z))

        return path

    def calculateEstimatedDistance(self):
        """
        Computes the total traveled distance based on the estimated trajectory.

        Returns
        -------
        total_distance : float
            The total traveled distance in meters.
        """
        total_distance = 0.0

        for i in range(1, len(self.trajectory)):
            prev_pose = self.trajectory[i - 1]
            curr_pose = self.trajectory[i]

            # Extract translation vectors
            prev_position = prev_pose[:3, 3]
            curr_position = curr_pose[:3, 3]

            # Compute Euclidean distance
            distance = np.linalg.norm(curr_position - prev_position)
            total_distance += distance

        return total_distance


vo = VisualOdometry(data_dir="dataset", real_time=False)



def print_calibration():
    print(vo.P_l)
    print(vo.K_l)
    print(vo.P_r)
    print(vo.K_r)


def print_ground_truth():
    print(vo.ground_truth)

# print_calibration()
#print_ground_truth()


estimated_paths = vo.estimate_path()

estimated_distance = vo.calculateEstimatedDistance()
print(vo.calculateDistanceGT(100000000))
print(estimated_distance)


