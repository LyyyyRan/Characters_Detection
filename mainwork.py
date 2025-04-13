# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:30:29 2022

@author: https://github.com/LyyyyRan
"""

# pkgs:
import time
import torch
import serial
from argparse import ArgumentParser
import cv2 as cv
from random import randint

# YOLO:
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# ignore some warnings:
import warnings

warnings.filterwarnings("ignore")


# UART:
def get_serial(serial_name='/dev/ttyUSB0', buardrate=115200):
    # Get object:
    uart = serial.Serial()

    # Config:
    uart.port = serial_name
    uart.buardrate = buardrate
    uart.bytesize = 8
    uart.stopbits = 1
    uart.parity = 'N'

    # open:
    uart.open()

    # return:
    return uart


# Send MSG:
def uart_sent(uart, MSG):
    # sent bytes one by one:
    for data in MSG:

        # uart must be opened:
        if not uart.isOpen():
            uart.open()

        # sent:
        uart.write(data)


# Configuration:
parser = ArgumentParser()

# Hardware:
parser.add_argument('--camera', default=0, help='Camera')
parser.add_argument('--use-serial', default=False, help='whether to use serial')
parser.add_argument('--serial', type=str, default='/dev/ttyUSB0',
                    help='USART: comX (in Windows) or /dev/ttyUSBx (in Linux)')
parser.add_argument('--buardrate', type=str, default='115200', help='Buardrate of USART')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

# about CV:
parser.add_argument('--weights', nargs='+', type=str, default='./weights/arabic_numbers.pt', help='{model}.pt')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.90, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.30, help='IoU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')

# for Debug:
parser.add_argument('--view-img', type=str, default=True,
                    help='whether to open a window to show img')  # cost a little time
parser.add_argument('--echo-result', type=str, default=False, help='whether to print results')  # cost little time
parser.add_argument('--draw-result', type=str, default=True, help='whether to draw BBoxes in images')  # cost much time
parser.add_argument('--echo-MSG', type=str, default=False,
                    help='whether to print the msg tobe sent')  # cost little time

# push to an opt:
opt = parser.parse_args()

# parameters:
weights, imgsz = opt.weights, opt.img_size
view_img = opt.view_img
echo_result = opt.echo_result
draw_result = opt.draw_result
echo_MSG = opt.echo_MSG

# print configuration:
if echo_result:
    print(opt)

# Device:
device = select_device(opt.device)

# Float 16 or not:
half = device.type != 'cpu'  # half precision only supported on CUDA

# Get Model:
model = attempt_load(weights, map_location=device)  # load FP32 model
print('**********************')
print('Get Model Completed!')
print('**********************')

# Get category names:
names = model.module.names if hasattr(model, 'module') else model.names
categories_num = len(names)

# Get category colors:
if draw_result:
    colors = [[randint(0, 255) for _ in range(3)] for _ in range(categories_num)]

# MSG Buffers to UART:
MSG_Header = [0Xff, 0Xfc, 0X00]
MSG_End = [0Xaa, 0Xcc]

# to Float 16:
if half:
    model.half()  # to FP16

# check image size:
imgsz = check_img_size(imgsz, s=model.stride.max())

# Get Camera:
while True:
    try:
        cap = cv.VideoCapture(opt.camera)

        # check camera:
        if cap.isOpened():
            print('Get camera completed!')
            print('**********************')
            break
        else:
            print('Failed to get camera, still trying...')

    except Exception as e:
        print(e)

# Get Serial:
while True:
    try:
        # Get Serial:
        uart = get_serial(opt.serial, opt.buardrate)

        # check camera:
        if uart.isOpen():
            print('Get serial completed!')
            print('**********************')
            break
        else:
            print('Failed to get serial, still trying...')

    except Exception as e:
        print(e)

    if not opt.use_serial:
        print('whether to use usart:', opt.use_serial)
        break
    else:
        print('whether to use usart:', opt.use_serial)

# Main work:
while True:
    # Get image from capture:
    flag, img = cap.read()

    # if succeed:
    if flag:
        # Lyyy: not necessary to resize
        # # resize to 640 * 640:
        # img = cv.resize(img, (640, 640))

        # Copy:
        im0 = img.copy()

        # to RGB:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # to Tensor:
        img = torch.from_numpy(img).to(device)

        # Float 16:
        img = img.half() if half else img.float()  # uint8 to fp16/32

        # Normalize: Squeeze value domain from [0, 255] to [0., 1.]
        img /= 255.0

        # Add a dimension batch size:
        if img.ndimension() != 4:
            img = img.unsqueeze(0)

        # [b, h, w, c] to [b, c, h, w]:
        img = img.permute(0, 3, 1, 2)

        # start time:
        start_time = time.time()

        # feed forward:
        pred = model(img, augment=opt.augment)[0]

        # NMS && to BBox:
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # end time:
        end_time = time.time()

        # detect time/s:
        detect_time = end_time - start_time

        # detect time/ms:
        detect_time *= 1000.

        # MSG Buffer to UART:
        MSG_Buffer = []

        # deal results from per image:
        for idx, det in enumerate(pred):
            # Get string:
            if echo_result:
                string = '%g: ' % idx
                string += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # print results:
                if echo_result:
                    # Add result:
                    for category in det[:, -1].unique():
                        num = (det[:, -1] == category).sum()  # detections per class
                        string += '%g %ss, ' % (num, names[int(category)])  # add to string

                    # Print results:
                    print(string + 'The time taken to detect: {:.0f} ms'.format(detect_time))

                # make msg to uart:
                for category in det[:, -1].unique():
                    # ToHex:
                    category_hex = int(category) + 1

                    # Add category to msg:
                    MSG_Buffer.append(category_hex)

                # Write results:
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image:
                    if draw_result:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # View image:
        if view_img:
            cv.imshow('view', im0)

        # Send Msg to UART:
        MSG_Header[2] = len(MSG_Buffer)

        # debug msg to uart:
        if echo_MSG:
            print('MSG to UART: ', end='')

            # Header:
            for data in MSG_Header:
                print(hex(data), end=' ')

            # Real Data:
            for data in MSG_Buffer:
                print(hex(data), end=' ')

            # End:
            for data in MSG_End:
                print(hex(data), end=' ')

            # just for print a '\n':
            print('')

        # if using serial:
        if opt.use_serial:
            # start send time:
            start_time = time.time()

            # Sent MSG:
            uart_sent(uart=uart, MSG=MSG_Header + MSG_Buffer + MSG_End)

            # end send time:
            end_time = time.time()

            if echo_result:
                print('The time taken to send msg: {:.2f} s'.format(end_time - start_time))

    else:
        break

    # sleep 1 ms to show image:
    if cv.waitKey(1) == ord(' '):
        print('Keyboard Interrupt!')
        break

# release && close windows:
cap.release()
cv.destroyAllWindows()

if __name__ == '__main__':
    pass
