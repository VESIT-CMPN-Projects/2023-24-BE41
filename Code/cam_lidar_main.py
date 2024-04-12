import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import sys

sys.path.append('/home/asus/project/Camerasidefinal')

from object_detection import object_detection_pipeline, load_object_detection_model, load_lane_detection_model, road_lines
#the quick brown fox jumped over lazy dog

############################
# Serial Functions

def read_tfluna_data():
    while True:
        counter = ser.in_waiting  # count the number of bytes waiting to be read
        bytes_to_read = 9
        if counter > bytes_to_read - 1:
            bytes_serial = ser.read(bytes_to_read)  # read 9 bytes
            ser.reset_input_buffer()  # reset buffer

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59:  # check first two bytes
                distance = bytes_serial[2] + bytes_serial[3] * 256  # distance in next two bytes
                strength = bytes_serial[4] + bytes_serial[5] * 256  # signal strength in next two bytes
                temperature = bytes_serial[6] + bytes_serial[7] * 256  # temp in next two bytes
                temperature = (temperature / 8) - 256  # temp scaling and offset
                return distance / 100.0, strength, temperature

def set_samp_rate(samp_rate=100):
    # change the sample rate
    samp_rate_packet = [0x5a, 0x06, 0x03, samp_rate, 00, 00]  # sample rate byte array
    ser.write(samp_rate_packet)  # send sample rate instruction
    return

def get_version():
    # get version info
    info_packet = [0x5a, 0x04, 0x14, 0x00]

    ser.write(info_packet)  # write packet
    time.sleep(0.1)  # wait to read
    bytes_to_read = 30  # prescribed in the product manual
    t0 = time.time()
    while (time.time() - t0) < 5:
        counter = ser.in_waiting
        if counter > bytes_to_read:
            bytes_data = ser.read(bytes_to_read)
            ser.reset_input_buffer()
            if bytes_data[0] == 0x5a:
                version = bytes_data[3:-1].decode('utf-8')
                print('Version -' + version)  # print version details
                return
            else:
                ser.write(info_packet)  # if fails, re-write packet
                time.sleep(0.1)  # wait

def set_baudrate(baud_indx=4):
    # get version info
    baud_hex = [[0x80, 0x25, 0x00],  # 9600
                [0x00, 0x4b, 0x00],  # 19200
                [0x00, 0x96, 0x00],  # 38400
                [0x00, 0xe1, 0x00],  # 57600
                [0x00, 0xc2, 0x01],  # 115200
                [0x00, 0x84, 0x03],  # 230400
                [0x00, 0x08, 0x07],  # 460800
                [0x00, 0x10, 0x0e]]  # 921600
    info_packet = [0x5a, 0x08, 0x06, baud_hex[baud_indx][0], baud_hex[baud_indx][1],
                   baud_hex[baud_indx][2], 0x00, 0x00]  # instruction packet

    prev_ser.write(info_packet)  # change the baud rate
    time.sleep(0.1)  # wait to settle
    prev_ser.close()  # close old serial port
    time.sleep(0.1)  # wait to settle
    ser_new = serial.Serial("/dev/serial0", baudrates[baud_indx], timeout=0)  # new serial device
    if ser_new.isOpen() == False:
        ser_new.open()  # open serial port if not open
    bytes_to_read = 8
    t0 = time.time()
    while (time.time() - t0) < 5:
        counter = ser_new.in_waiting
        if counter > bytes_to_read:
            bytes_data = ser_new.read(bytes_to_read)
            ser_new.reset_input_buffer()
            if bytes_data[0] == 0x5a:
                indx = [ii for ii in range(0, len(baud_hex)) if
                        baud_hex[ii][0] == bytes_data[3] and
                        baud_hex[ii][1] == bytes_data[4] and
                        baud_hex[ii][2] == bytes_data[5]]
                print('Set Baud Rate = {0:1d}'.format(baudrates[indx[0]]))
                time.sleep(0.1)
                return ser_new
            else:
                ser_new.write(info_packet)  # try again if wrong data received
                time.sleep(0.1)  # wait 100ms
                continue

############################
# Configurations

baudrates = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]  # baud rates
prev_indx = 4  # previous baud rate index (current TF-Luna baudrate)
prev_ser = serial.Serial("/dev/serial0", baudrates[prev_indx], timeout=0)  # mini UART serial device
if prev_ser.isOpen() == False:
    prev_ser.open()  # open serial port if not open
baud_indx = 4  # baud rate to be changed to (new baudrate for TF-Luna)
ser = set_baudrate(baud_indx)  # set baudrate, get new serial at new baudrate
set_samp_rate(100)  # set sample rate 1-250
get_version()  # print version info for TF-Luna

##############################################
# Plotting functions

def plotter():
    plt.style.use('ggplot')  # plot formatting
    fig, axs = plt.subplots(1, 1, figsize=(5, 8))  # create figure
    fig.subplots_adjust(wspace=0.05)

    axs.set_xlim([-1.0, 1.0])  # strength bar width
    axs.set_xticks([])  # remove x-ticks
    axs.set_ylim([1.0, 2**16])  # set signal strength limits
    axs.yaxis.tick_right()  # move strength ticks to the right
    axs.yaxis.set_label_position('right')  # label to the right
    axs.set_ylabel('Signal Strength', fontsize=16, labelpad=6.0)
    axs.set_yscale('log')  # log scale for better visual
    # draw and background specification
    fig.canvas.draw()  # draw initial plot

    ax2_bgnd = fig.canvas.copy_from_bbox(axs.bbox)  # get background

    bar1, = axs.bar(0.0, 1.0, width=1.0, color=plt.cm.Set1(2))
    fig.show()  # show plot
    return fig, axs, ax2_bgnd, bar1

def plot_updater(fig, axs, ax2_bgnd, bar1, strength):
    fig.canvas.restore_region(ax2_bgnd)  # restore background

    bar1.set_height(strength)  # update signal strength
    if strength < 100.0 or strength > 30000.0:
        bar1.set_color(plt.cm.Set1(0))  # if invalid strength, make the bar red
    else:
        bar1.set_color(plt.cm.Set1(2))  # green bar

    axs.draw_artist(bar1)  # draw signal strength bar
    fig.canvas.blit(axs.bbox)  # blitting
    fig.canvas.flush_events()  # required for blitting
    return bar1

##############################################
# Tkinter GUI
def blend_images(image1, image2, opacity):
    # Resize the images to have the same dimensions
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    scaled_width2 = width1 // 4;
    scaled_height2 = int(height2 * scaled_width2 / width2)
    image2_resized = cv2.resize(image2, (scaled_width2, scaled_height2))

    # Calculate the position to overlay the second image
    overlay_x = width1 - scaled_width2
    overlay_y = height1 - scaled_height2  # Adjust this value to change the vertical offset
    

    # Blend the images using weighted addition
    blended_image = image1.copy()
    blended_image[overlay_y:overlay_y+scaled_height2, overlay_x:overlay_x+scaled_width2] = cv2.addWeighted(image1[overlay_y:overlay_y+scaled_height2, overlay_x:overlay_x+scaled_width2], opacity, image2_resized, 1 - opacity, 0)
    return blended_image

class VideoApp:
    def _init_(self, window, window_title, output_video_path):
        self.window = window
        self.window.title(window_title)

        # Load the object detection model
        self.model = load_object_detection_model()
        self.lanemodel = load_lane_detection_model()

        self.video_source = cv2.VideoCapture("testvideos/final_testvids/final_testvid2.mp4") #"testvideos/trynew1.mp4"
        self.out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), int(self.video_source.get(5)),
                                   (int(720), int(480)))
        self.canvas = tk.Canvas(window, width=int(800), height=int(window.winfo_screenheight() * 0.6)) #window.winfo_screenwidth() * 0.8
        self.canvas.grid(row=0, column=0, rowspan=6)
        
        self.label_distance = tk.Label(window, text="Distance: ")
        self.label_distance.grid(row=5,column=0)

        self.btn_quit = tk.Button(window, text="Quit", width=5, command=self.on_quit)
        self.btn_quit.grid(row=6, column=0)

        self.plot_pts = 100
        self.fig, self.axs, self.ax2_bgnd, self.bar1 = plotter()
        self.dist_array = []

        self.update()
        self.window.mainloop()

    # def update(self):
        # ret, frame = self.video_source.read()
        # if ret:
            # processed_frame = object_detection_pipeline(frame, self.model)
            # processed_frame = cv2.resize(processed_frame, (1280,720))
            # processed_frame = road_lines(processed_frame, self.lanemodel)
            # processed_frame = cv2.resize(processed_frame,(640, 360))
            
            # #process your distances first, if conditions hit
            # image2 = cv2.imread('alert1.png')
            # image2 = cv2.resize(image2, (64, 64))
            
            # processed_frame = blend_images(processed_frame, image2, 0.5)
            # self.photo = self.convert_to_tk_image(processed_frame)
            # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            # self.out.write(processed_frame)

            # distance, strength, temperature = read_tfluna_data()  # read values
            # print("\tdistance : ", distance)
            # print("\tstrength : ", strength)
            # self.label_distance.config(text="Distance: {:.2f}".format(distance))

            # self.dist_array.append(distance)  # append to array
            # if len(self.dist_array) > self.plot_pts:
                # self.dist_array = self.dist_array[1:]  # drop the first point (maintain array size)
                # self.bar1 = plot_updater(self.fig, self.axs, self.ax2_bgnd, self.bar1, strength)  # update plot

        # self.window.after(10, self.update)
        
    
    def update(self):
        ret, frame = self.video_source.read()
        if ret:
            processed_frame = object_detection_pipeline(frame, self.model)
            processed_frame = cv2.resize(processed_frame, (1280,720))
            processed_frame = road_lines(processed_frame, self.lanemodel)
            processed_frame = cv2.resize(processed_frame,(720, 480))
            
            distance, strength, temperature = read_tfluna_data()
        
            if 0.00 <= distance <= 0.40 :
                image2 = cv2.imread('alert1.png')
                image2 = cv2.resize(image2, (64, 64))
                
                processed_frame = blend_images(processed_frame, image2, 0.5)

            
            self.photo = self.convert_to_tk_image(processed_frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.out.write(processed_frame)
            self.label_distance.config(text="Distance: {:.2f}".format(distance))
            self.dist_array.append(distance)
            
            # Update the plot if the array exceeds the plot points
            if len(self.dist_array) > self.plot_pts:
                self.dist_array = self.dist_array[1:]
                self.bar1 = plot_updater(self.fig, self.axs, self.ax2_bgnd, self.bar1, strength)
                
        self.window.after(10, self.update)


    def convert_to_tk_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        photo = ImageTk.PhotoImage(image=img)
        return photo

    def on_quit(self):
        self.video_source.release()
        self.out.release()
        self.window.destroy()

if _name_ == "_main_":
    root = tk.Tk()
    app = VideoApp(root, "Video Processing App", 'newvideo.mp4')