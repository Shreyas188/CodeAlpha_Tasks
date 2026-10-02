"""
SORT: Simple Online and Realtime Tracking
A minimal implementation of SORT tracking algorithm.
Original paper: Bewley et al. 2016
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two bounding box arrays.
    Boxes in [x1,y1,x2,y2] format.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    iou = wh / (area_test + area_gt - wh + 1e-10)
    return iou


def convert_bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] to [cx,cy,s,r] for Kalman filter."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h) if h != 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Convert [cx,cy,s,r] back to [x1,y1,x2,y2]."""
    w = np.sqrt(abs(x[2]) * abs(x[3]))
    h = abs(x[2]) / (w + 1e-10)
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """Tracks a single object using Kalman Filter."""
    count = 0

    def __init__(self, bbox):
        # State: [cx, cy, s, r, vx, vy, vs]
        # Measurement: [cx, cy, s, r]
        self.kf_F = np.eye(7, 7)  # Transition matrix
        self.kf_H = np.eye(4, 7)  # Measurement matrix
        self.kf_R = np.eye(4) * 10  # Measurement noise
        self.kf_Q = np.eye(7)  # Process noise
        self.kf_P = np.eye(7) * 10  # Covariance
        
        # Velocity in state transition
        for i in range(3):
            self.kf_F[i, i+4] = 1
        
        self.kf_Q[4:, 4:] *= 0.01
        self.kf_Q[2, 2] *= 0.01
        self.kf_R[2:, 2:] *= 10
        self.kf_P[4:, 4:] *= 1000
        self.kf_P *= 10
        
        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(bbox)
        
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
    
    def update(self, bbox):
        """Update tracker with new bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Kalman update step
        z = convert_bbox_to_z(bbox)
        y = z - self.kf_H @ self.x
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ self.kf_H
        self.kf_P = I_KH @ self.kf_P
    
    def predict(self):
        """Advance the state vector and return predicted bounding box."""
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        # Kalman predict step
        self.x = self.kf_F @ self.x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.x))
        return self.history[-1]
    
    def get_state(self):
        """Return current bounding box estimate."""
        return convert_x_to_bbox(self.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to tracked objects using Hungarian algorithm.
    Returns:
        matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack([row_ind, col_ind], axis=1)
    else:
        matched_indices = np.empty((0, 2), dtype=int)
    
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT Multi-Object Tracker.
    
    Parameters:
        max_age     : Maximum frames a track can be missing before deletion.
        min_hits    : Minimum detections before a track is confirmed.
        iou_threshold: Minimum IoU for detection-to-track assignment.
    """
    
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
    
    def update(self, dets=np.empty((0, 5))):
        """
        Update trackers with new detections.
        
        Args:
            dets: numpy array of shape (N, 5) with [x1,y1,x2,y2,score]
        
        Returns:
            numpy array of shape (M, 5) with [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing Kalman filters
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4] if len(dets) > 0 else np.empty((0, 4)),
            trks[:, :4],
            self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
        
        # Build output — only confirmed, recently-updated tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
