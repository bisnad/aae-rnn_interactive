import torch
from torch import nn
import numpy as np
import torch.nn.functional as nnF

from common.quaternion import qmul, qrot, qnormalize_np
from common.quaternion_torch import slerp, qfix

config = {"skeleton": None,
          "model_encoder": None,
          "model_decoder": None,
          "device": "cuda",
          "seq_window_length": 8,
          "seq_window_overlap": 2,
          "orig_seq": None
          }

def smooth_1d(data_1d, window_length, window_type="hanning"):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is padded with zeros at both ends
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """
    
    if data_1d.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if data_1d.size < window_length:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_length<3:
        return data_1d

    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #print("data_1d s ", data_1d.shape)
    
    pad_left = np.flip(data_1d[:(window_length - 1)//2])
    pad_right = np.flip(data_1d[-(window_length - 1)//2:])
    
    data_padded=np.concatenate((pad_left, data_1d, pad_right))
    
    #print("data_padded s ", data_padded.shape)
    
    #print(len(s))
    if window_type == 'flat': #moving average
        window=np.ones(window_length,'d')
    else:
        window=eval('np.'+window_type+'(window_length)')

    data_smooth=np.convolve(window/window.sum(),data_padded,mode='valid')
    
    #print("data_smooth s ", data_smooth.shape)
    
    return data_smooth            
  
def smooth(data, window_length, window_type="hanning"):
    """
    helper function for multidimensional data
    """
    
    orig_shape = data.shape

    data = np.reshape(data, (orig_shape[0], -1))
    data_dim = data.shape[1]
    data_smooth = [ smooth_1d(data[:, d], window_length, window_type) for d in range(data_dim) ]
    data_smooth = np.stack(data_smooth, axis=1)
    data_smooth = np.reshape(data_smooth, orig_shape)

    return data_smooth

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model_encoder = config["model_encoder"]
        self.model_decoder = config["model_decoder"]
        self.device = config["device"]
        self.seq_window_length = config["seq_window_length"]
        self.seq_window_overlap = config["seq_window_overlap"]
        self.orig_seq = config["orig_seq"]
        
        self.seq_window_offset = self.seq_window_length - self.seq_window_overlap
        self.seq_length = self.orig_seq.shape[0]
        self.joint_count = self.orig_seq.shape[1]
        self.joint_dim = self.orig_seq.shape[2]
        self.pose_dim = self.joint_count * self.joint_dim
        
        self.joint_offsets = self.skeleton ["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton ["parents"]
        self.joint_children = self.skeleton ["children"]
        
        self._create_edge_list()
        
        self.orig_seq_frame_index1 = self.seq_window_offset
        self.orig_seq_frame_index2 = self.seq_window_offset
        
        self.orig_seq_frame_incr1 = self.seq_window_offset
        self.orig_seq_frame_incr2 = self.seq_window_offset
        self.orig_seq_frame_range1 = [0, self.seq_length - self.seq_window_length]
        self.orig_seq_frame_range2 = [0, self.seq_length - self.seq_window_length]
        
        self.orig_encoding_mix_factor = 0.0
        self.encoding_offset = torch.zeros((1, self.model_encoder.latent_dim)).to(self.device)
        
        #self.gen_seq = torch.from_numpy(self.orig_seq[:self.seq_window_length, ...]).to(self.device)
        self.gen_seq = torch.Tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.seq_window_length, self.joint_count, 1).to(self.device)

        self.gen_seq_window = None
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
        self.seq_update_index = 0
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setFrameIndex1(self, index):
        self.orig_seq_frame_index1 = min(index, self.seq_length - self.seq_window_length)

    def setFrameIndex2(self, index):
        self.orig_seq_frame_index2 = min(index, self.seq_length - self.seq_window_length)
           
    def setFrameRange1(self, startFrame, endFrame):
        self.orig_seq_frame_range1[0] = min(startFrame, self.seq_length - self.seq_window_length)
        self.orig_seq_frame_range1[1] = min(endFrame, self.seq_length - self.seq_window_length)
        
    def setFrameRange2(self, startFrame, endFrame):
        self.orig_seq_frame_range2[0] = min(startFrame, self.seq_length - self.seq_window_length)
        self.orig_seq_frame_range2[1] = min(endFrame, self.seq_length - self.seq_window_length)
        
    def setFrameIncrement1(self, incr):
        self.orig_seq_frame_incr1 = incr
        
    def setFrameIncrement2(self, incr):
        self.orig_seq_frame_incr2 = incr
    
    def setEncodingMixFactor(self, factor):
        self.orig_encoding_mix_factor = factor
        
    def setEncodingOffset(self, offset):
        self.encoding_offset = torch.tensor(offset, dtype=torch.float32).to(self.device)
                
    def update(self):
        

        #print("self.seq_update_index ", self.seq_update_index)
        
        # generate next skel pose
        pred_pose = self.gen_seq[self.seq_update_index, ...]

        pred_pose = pred_pose.reshape((-1, 4))
        pred_pose = nn.functional.normalize(pred_pose, p=2, dim=1)
        pred_pose = pred_pose.reshape((1, self.joint_count, self.joint_dim))
        
        zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
        zero_trajectory = zero_trajectory.to(self.device)
        
        self.synth_pose_wpos, self.synth_pose_wrot = self._forward_kinematics(torch.unsqueeze(pred_pose,dim=0), zero_trajectory)
        
        self.synth_pose_wpos = self.synth_pose_wpos.detach().cpu().numpy()
        self.synth_pose_wpos = self.synth_pose_wpos.reshape((self.joint_count, 3))
        
        self.synth_pose_wrot = self.synth_pose_wrot.detach().cpu().numpy()
        self.synth_pose_wrot = self.synth_pose_wrot.reshape((self.joint_count, 4))
        
        self.seq_update_index += 1
        
        if self.seq_update_index >= self.seq_window_offset:
            
            #print("_gen")
            self._gen()
            self._blend()

            self.seq_update_index = 0
        

    def _gen(self):
         
        # encode orig seq window 1 and orig seq window 2
        
        #print("orig seq1 excerpt from ", self.orig_seq_frame_index1, " to ", (self.orig_seq_frame_index1 + self.seq_window_length) )
        
        orig_seq_window1 = self.orig_seq[self.orig_seq_frame_index1:self.orig_seq_frame_index1 + self.seq_window_length, ...]
        orig_seq_window1 = torch.from_numpy(orig_seq_window1).reshape((1, self.seq_window_length, self.pose_dim)).to(self.device)
        
        #print("orig seq2 excerpt from ", self.orig_seq_frame_index2, " to ", (self.orig_seq_frame_index2 + self.seq_window_length) )

        orig_seq_window2 = self.orig_seq[self.orig_seq_frame_index2:self.orig_seq_frame_index2 + self.seq_window_length, ...]
        orig_seq_window2 = torch.from_numpy(orig_seq_window2).reshape((1, self.seq_window_length, self.pose_dim)).to(self.device)
        
        with torch.no_grad():
            encoding1 = self.model_encoder(orig_seq_window1)
            encoding2 = self.model_encoder(orig_seq_window2)

        # mix encoding 1 and encoding 2
        encoding = encoding1 * (1.0 - self.orig_encoding_mix_factor) + encoding2 * self.orig_encoding_mix_factor
        
        # add encoding offset
        encoding += self.encoding_offset 
        
        # decode encoding
        with torch.no_grad():
            self.gen_seq_window = self.model_decoder(encoding)
            
        self.gen_seq_window = self.gen_seq_window.reshape(self.seq_window_length, self.joint_count, self.joint_dim)
        
        self.gen_seq_window = nnF.normalize(self.gen_seq_window , dim=2)
        self.gen_seq_window = qfix(self.gen_seq_window)
        
        # increment orig frame index 1 and orig frame index 2
        self.orig_seq_frame_index1 += self.orig_seq_frame_incr1
        if self.orig_seq_frame_index1 < self.orig_seq_frame_range1[0]:
            self.orig_seq_frame_index1 = self.orig_seq_frame_range1[0]  
        elif self.orig_seq_frame_index1 >= self.orig_seq_frame_range1[1]:
            self.orig_seq_frame_index1 = self.orig_seq_frame_range1[0]  
            
        self.orig_seq_frame_index2 += self.orig_seq_frame_incr2
        if self.orig_seq_frame_index2 < self.orig_seq_frame_range2[0]:
            self.orig_seq_frame_index2 = self.orig_seq_frame_range2[0]  
        elif self.orig_seq_frame_index2 >= self.orig_seq_frame_range2[1]:
            self.orig_seq_frame_index2 = self.orig_seq_frame_range2[0]  
      

    def _blend(self):
        
        # roll seq window
        self.gen_seq = torch.roll(self.gen_seq, -self.seq_window_offset, 0)    
        
        # blend overlap region between gen_seq and gen_seq_window
        #blend_slope = torch.linspace(0.0, ((self.seq_window_overlap - 1) / self.seq_window_overlap), self.seq_window_overlap).unsqueeze(1).repeat(1, self.joint_count).to(self.device)
        
        blend_slope = torch.linspace(0.0, 1.0, self.seq_window_overlap).unsqueeze(1).repeat(1, self.joint_count).to(self.device)
        
        #print("blend_slope s ", blend_slope.shape)
        #print("blend_slope ", blend_slope)
        
        #print("self.gen_seq s ", self.gen_seq.shape)
        #print("self.gen_seq_window s ", self.gen_seq_window.shape)
        
        self.gen_seq = qfix(self.gen_seq)
        self.gen_seq_window = qfix(self.gen_seq_window)

        blend_seq = slerp(self.gen_seq[:self.seq_window_overlap].reshape(-1, 4), self.gen_seq_window[:self.seq_window_overlap].reshape(-1, 4), blend_slope.reshape(-1))
        blend_seq = blend_seq.reshape(self.seq_window_overlap, self.joint_count, self.joint_dim)
        
        blend_seq = torch.from_numpy(smooth(blend_seq.detach().cpu().numpy(), 5)).to(self.device)
        blend_seq = nnF.normalize(blend_seq , dim=2)
        blend_seq = qfix(blend_seq)
    
        #print("blend_seq s ", blend_seq.shape)

        self.gen_seq[:self.seq_window_overlap] = blend_seq
        self.gen_seq[self.seq_window_overlap:] = torch.clone(self.gen_seq_window[self.seq_window_overlap:])
        
        self.gen_seq = qfix(self.gen_seq)

        
    def _forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4
        
        toffsets = torch.tensor(self.joint_offsets).to(self.device)
        
        positions_world = []
        rotations_world = []

        expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], self.joint_offsets.shape[0], self.joint_offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for jI in range(self.joint_offsets.shape[0]):
            if self.joint_parents[jI] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self.joint_parents[jI]], expanded_offsets[:, :, jI]) \
                                       + positions_world[self.joint_parents[jI]])
                if len(self.joint_children[jI]) > 0:
                    rotations_world.append(qmul(rotations_world[self.joint_parents[jI]], rotations[:, :, jI]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]).to(self.device))

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)
        