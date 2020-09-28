import numpy as np
import matplotlib.pyplot as plt
import signal

class Plotter:
    def __init__(self, optimize=False):
        # Init empty figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.autoscale(False)
        self.ax.view_init(elev=30, azim=-120)

        # Variables to facilitate update
        self.drawn = False
        self.links = None
        self.joints = None
        self.joints_frames = []
        self.tool_frame = []

        self.optimize = optimize

        self.quiver_len = 0.15
        self.colors = ["red", "green", "blue"]

        plt.ion()
        plt.show()

        # Set the signal handler and a 0.1 second plot updater
        signal.signal(signal.SIGALRM, self._plot_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.2, 0.2)

    def _plot_handler(self, sig, frame):
        try:
            plt.pause(0.001)
        except(AttributeError):
            pass

    def update_boundary(self, pos):
        # Calculate maximum and minimum positions
        x_max = pos[0, :].max()
        x_min = pos[0, :].min()
        y_max = pos[1, :].max()
        y_min = pos[1, :].min()
        z_max = pos[2, :].max()
        z_min = pos[2, :].min()

        if self.optimize == "O2":
            all_max = np.absolute(np.array((x_max, x_min, y_max, y_min, z_max, z_min) )).max()

            if all_max > self.ax.get_xbound()[1]:
                self.ax.set_xbound( -(all_max + self.quiver_len), (all_max + self.quiver_len) )
                self.ax.set_ybound( -(all_max + self.quiver_len), (all_max + self.quiver_len) )
                if abs(z_min) < 1e-3:
                    self.ax.set_zbound(0, 2*(all_max + self.quiver_len))
                else:
                    self.ax.set_zbound(z_min-self.quiver_len, z_min+all_max+self.quiver_len)

        else:    
            # Calculate largest needed boundary
            max_range = np.array( [x_max-x_min, y_max-y_min, z_max-z_min] ).max()

            self.ax.set_xbound( (x_max+x_min)/2 - max_range/2 -self.quiver_len, (x_max+x_min)/2 + max_range/2 + self.quiver_len )
            self.ax.set_ybound( (y_max+y_min)/2 - max_range/2 -self.quiver_len, (y_max+y_min)/2 + max_range/2 + self.quiver_len )
            if abs(z_min) < 1e-3 :
                self.ax.set_zbound(0, max_range + 2*self.quiver_len)
            else:
                self.ax.set_zbound( (z_max+z_min)/2 - max_range/2 -self.quiver_len, (z_max+z_min)/2 + max_range/2 + self.quiver_len )

    def start(self, base, pose_all, t):
        self.drawn = True

        # Include origin
        pos = np.zeros([3, len(pose_all)+3])

        pos[:, 1] = base[:3, 3]
        pos[:, 2:-1] = np.transpose(pose_all[:, :3, 3])
        pos[:, -1] = t[:3, 3]

        self.update_boundary(pos)

        # Plot links, saving reference for future updates
        self.links, = self.ax.plot(pos[0, :], pos[1, :], pos[2, :],
                                  linewidth=2, color="black")

        # Plot joint frames origins
        self.joints = self.ax.scatter(pose_all[:, 0, 3], # xs
                                      pose_all[:, 1, 3], # ys
                                      pose_all[:, 2, 3], # zs
                                      c="gray", marker="s", s=50) 

        if not self.optimize or self.optimize == "O0":
            # Plot joint frames
            for i in range(len(pose_all)):
                self.joints_frames.append([])

                # Get frame in global coordinate system
                frame = np.matmul(pose_all[i][:3, :3], [[self.quiver_len, 0,   0], 
                                                        [0,   self.quiver_len, 0], 
                                                        [0,   0,   self.quiver_len]])

                for j in range(3):
                    self.joints_frames[i].append(self.ax.quiver(pose_all[i][0][3], # x
                                                                pose_all[i][1][3], # y
                                                                pose_all[i][2][3], # z
                                                                frame[0][j], # dx
                                                                frame[1][j], # dy
                                                                frame[2][j], # dz
                                                                linewidth=2, color=self.colors[j]) )

        frame = np.matmul(t[:3, :3], [[self.quiver_len, 0,   0], 
                                      [0,   self.quiver_len, 0], 
                                      [0,   0,   self.quiver_len]])

        label = ['x', 'y', 'z']
        # Plot tool frame
        for i in range(3):
            self.tool_frame.append(self.ax.quiver(t[0][3], t[1][3], t[2][3],
                                                  frame[0][i], frame[1][i], frame[2][i],
                                                  linewidth=2, color=self.colors[i], label=label[i]))

        self.ax.legend()

    def plot(self, base, pose_all, t):
        if(not self.drawn):
            self.start(base, pose_all, t)
            return

        # Include origin
        pos = np.zeros([3, len(pose_all)+3])

        pos[:, 1] = base[:3, 3]
        pos[:, 2:-1] = np.transpose(pose_all[:, :3, 3])
        pos[:, -1] = t[:3, 3]

        self.update_boundary(pos)

        # Update links
        self.links.set_xdata(pos[0, :])
        self.links.set_ydata(pos[1, :])
        self.links.set_3d_properties(pos[2, :])

        # Update joints positions
        self.joints._offsets3d = (pose_all[:, 0, 3], # xs
                                  pose_all[:, 1, 3], # ys
                                  pose_all[:, 2, 3]) # zs

        if not self.optimize or self.optimize == "O0":
            # Update joint frames
            for i in range(len(pose_all)):
                # Get frame in global coordinate system
                frame = np.matmul(pose_all[i][:3, :3], [[self.quiver_len, 0,   0], 
                                                        [0,   self.quiver_len, 0], 
                                                        [0,   0,   self.quiver_len]])

                for j in range(3):
                    self.joints_frames[i][j].set_segments([[[pose_all[i][0][3], 
                                                             pose_all[i][1][3], 
                                                             pose_all[i][2][3]],
                                                            [frame[0][j]+pose_all[i][0][3], 
                                                             frame[1][j]+pose_all[i][1][3],
                                                             frame[2][j]+pose_all[i][2][3]]]])

        frame = np.matmul(t[:3, :3], [[self.quiver_len, 0,   0], 
                                [0,   self.quiver_len, 0], 
                                [0,   0,   self.quiver_len]])

        # Update tool frame
        for i in range(3):
            self.tool_frame[i].set_segments([[[t[0][3], t[1][3], t[2][3]],
                                              [frame[0][i]+t[0][3], frame[1][i]+t[1][3], frame[2][i]+t[2][3]] ]])