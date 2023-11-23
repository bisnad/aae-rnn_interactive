import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "latent_dim": 32,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.latent_dim = config["latent_dim"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/frameindex1", self.setFrameIndex1)
        self.dispatcher.map("/mocap/frameindex2", self.setFrameIndex2)
        
        self.dispatcher.map("/mocap/framerange1", self.setFrameRange1)
        self.dispatcher.map("/mocap/framerange2", self.setFrameRange2)
        
        self.dispatcher.map("/mocap/frameincr1", self.setFrameIncrement1)
        self.dispatcher.map("/mocap/frameincr2", self.setFrameIncrement2)        
        
        self.dispatcher.map("/synth/encodingmixfactor", self.setEncodingMixFactor)
        self.dispatcher.map("/synth/encodingoffset", self.setEncodingOffset)    
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setFrameIndex1(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setFrameIndex1(index)

    def setFrameIndex2(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setFrameIndex2(index)
        
    def setFrameRange1(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setFrameRange1(startFrame, endFrame)

    def setFrameRange2(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setFrameRange2(startFrame, endFrame)    

    def setFrameIncrement1(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setFrameIncrement1(incr)

    def setFrameIncrement2(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setFrameIncrement2(incr)
        
    def setEncodingMixFactor(self, address, *args):
        
        factor = args[0]
        
        self.synthesis.setEncodingMixFactor(factor)
        
    def setEncodingOffset(self, address, *args):
        
        offset = []
        
        for d in range(self.latent_dim):
        
            offset.append(args[d])
        
        self.synthesis.setEncodingOffset(offset)

