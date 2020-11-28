#!/usr/bin/python
# -*- coding: UTF-8 -*-
import ctypes
import inspect

import numpy
import math
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QThread, QMutex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from track import *
from yolov5.utils.general import check_img_size


class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.s_rtsp = "rtsp://iscas:opqwer12@192.168.100.176:554/Streaming/Channels/101"
        self.cap = cv2.VideoCapture()
        self.queue = Queue()
        self.qmut_1 = QMutex()
        self.origin_coord = [960, 540]
        self.func_weight = [1, 0, 0]
        self.product_thread = ProductThread(cap=self.cap)
        self.consume_thread = ConsumeThread(qmut_1=self.qmut_1, queue=self.queue)
        self.paint_thread = PaintLineThread(qmut_1=self.qmut_1, origin_coord=self.origin_coord,
                                            func_weight=self.func_weight, queue=self.queue)
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.open_close = False
        self.go_person = 0
        self.come_person = 0
        self.dis_id = {}

        self.warn_count = 0

    def set_ui(self):

        self.__layout_main = QHBoxLayout()
        self.__layout_fun_button = QVBoxLayout()
        self.__layout_data_show = QVBoxLayout()

        # 添加右侧的统计展示控件，主要是想通过Button来展示
        self.__layout_statistic = QVBoxLayout()
        self.text_sum_person = QLabel()
        self.button_sum_person = QPushButton(u'0')
        self.text_go_person = QLabel()
        self.button_go_person = QPushButton(u'0')
        self.text_come_person = QLabel()
        self.text_come_alarm = QLabel()
        self.button_come_person = QPushButton(u'0')
        self.text_sum_person.setText(u'当前跟踪到的总人数：')
        self.button_sum_person.setEnabled(False)
        self.text_go_person.setText(u'出去的人数')
        self.button_go_person.setEnabled(False)
        self.text_come_person.setText(u'回来的人数')
        self.button_come_person.setEnabled(False)
        self.text_come_alarm.setText(u'')


        self.button_open_camera = QPushButton(u'打开相机')

        self.button_detect = QPushButton(u'开始检测')

        self.button_close = QPushButton(u'退出')

        # Button 的颜色修改
        button_color = [self.button_open_camera, self.button_detect, self.button_close]
        for i in range(len(button_color)):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:rgb(78,255,255)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
        self.button_detect.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        # 设置button的样式
        button_static_list = [self.button_sum_person, self.button_go_person, self.button_come_person]
        for button in button_static_list:
            button.setMinimumHeight(50)
            button.setStyleSheet("QPushButton{font-family:'宋体';font-size:16px;color:rgb(0,0,0,255);}")

            # move()方法移动窗口在屏幕上的位置到x = 500，y = 500坐标。
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QLabel()
        self.label_move = QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(700, 600)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_detect)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        # 布局里面添加控件

        self.__layout_statistic.addWidget(self.text_sum_person)
        self.__layout_statistic.addWidget(self.button_sum_person)
        self.__layout_statistic.addWidget(self.text_go_person)
        self.__layout_statistic.addWidget(self.button_go_person)
        self.__layout_statistic.addWidget(self.text_come_person)
        self.__layout_statistic.addWidget(self.button_come_person)
        self.__layout_statistic.addWidget(self.text_come_alarm)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)
        self.__layout_main.addLayout(self.__layout_statistic)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'摄像头')

        # 设置背景图片
        # palette1 = QPalette()
        # palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('1.png')))
        # self.setPalette(palette1)

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.product_thread.sinOut.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)
        self.button_detect.clicked.connect(self.button_detect_click)
        self.paint_thread.detOut.connect(self.show_camera)
        self.consume_thread.sum_person.connect(self.show_sum_person)
        self.consume_thread.bbox_id.connect(self.show_go_come_person)

    def show_go_come_person(self, bbox_id):

        if self.open_close:
            bboxs = bbox_id[0]
            warn_flag = False
            ids = bbox_id[1]
            for i in range(len(bboxs)):
                x_center = (bboxs[i][0] + bboxs[i][2]) / 2 - self.origin_coord[0]
                y_center = self.origin_coord[1] - (bboxs[i][1] + bboxs[i][3]) / 2

                dis = x_center * self.func_weight[0] + y_center * self.func_weight[1] + self.func_weight[2]

                if ids[i] not in self.dis_id.keys():
                    if dis > 0:
                        flag = 1
                    else:
                        flag = 0
                    self.dis_id[ids[i]] = [flag, 0]
                else:
                    f = self.dis_id[ids[i]][0]
                    if f == 1:  # come
                        if dis < -10:
                            self.go_person += 1
                            self.dis_id.pop(ids[i])
                    else:
                        if dis > 10:
                            self.come_person += 1
                            warn_flag = True
                            self.warn_count = 0
                            self.text_come_alarm.setText(u'Waring! Someone Comes!')
                            self.dis_id.pop(ids[i])
            # 清理dict中已经不再监控范围内的数据

            for ide in list(self.dis_id.keys()):
                if ide not in ids:
                    count = self.dis_id[ide][1]
                    count += 1
                    if count > 3:
                        self.dis_id.pop(ide)
                    else:
                        self.dis_id[ide] = [self.dis_id[ide][0], count]
            if not warn_flag:
                self.warn_count += 1
            if self.warn_count >= 50:
                self.text_come_alarm.setText(u'')
                self.warn_count = 0
            self.button_go_person.setText(str(self.go_person))
            self.button_come_person.setText(str(self.come_person))

    def show_sum_person(self, num):
        if self.open_close:
            self.button_sum_person.setText(str(num))

    def button_detect_click(self):
        if not self.open_close:
            self.consume_thread.begin()
            self.consume_thread.start()
            self.open_close = True
            self.paint_thread.begin()
            self.paint_thread.start()
            self.button_detect.setText(u'停止检测')
        else:
            print("Close The Camera !!")
            self.open_close = False
            self.paint_thread.stop()
            self.paint_thread.quit()
            self.paint_thread.wait()
            self.consume_thread.stop()
            self.consume_thread.quit()
            self.consume_thread.wait()
            self.queue.join()
            self.label_show_camera.clear()
            self.button_detect.setText(u'开始检测')
            self.button_sum_person.setText(u'')
            self.button_go_person.setText(u'')
            self.button_come_person.setText(u'')

    def button_open_camera_click(self):
        if not self.open_close:
            print("Opening The Camera !!!!")
            flag = self.cap.open(self.s_rtsp)
            # flag = self.cap.isOpened()
            if not flag:
                QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                              buttons=QMessageBox.Ok,
                                              defaultButton=QMessageBox.Ok)
            else:
                self.product_thread.begin()
                self.product_thread.start()
                self.open_close = True
                self.button_open_camera.setText(u'关闭相机')
        else:
            print("Close The Camera !!")
            self.open_close = False
            self.product_thread.stop()
            print("step 1111111111111")
            self.cap.release()
            print("step 22222222222222")
            self.label_show_camera.clear()
            print("step 333333333333333")
            self.button_open_camera.setText(u'打开相机')

    def show_camera(self, image):

        # image = self.get_line(image)
        if self.open_close:
            show = cv2.resize(image, (640, 480))

            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            # print(show.shape[1], show.shape[0])
            # show.shape[1] = 640, show.shape[0] = 480
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))
        else:
            print("The Camera is Closed!!")

    def closeEvent(self, event):
        ok = QPushButton()
        cacel = QPushButton()

        msg = QMessageBox(QMessageBox.Warning, u"关闭", u"确定退出？")

        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cacel, QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()

            event.accept()


class PaintLineThread(QThread):
    detOut = pyqtSignal(numpy.ndarray)

    def __init__(self, qmut_1, origin_coord, func_weight, queue, parent=None):
        super(PaintLineThread, self).__init__(parent)
        self.qmut_1 = qmut_1
        self.queue = queue
        self.Painting = False
        self.origin_coord = origin_coord
        self.func_weight = func_weight

    def cul_line_func(self, flag, val):
        if self.func_weight[0] == 0 and self.func_weight[1] == 0:
            return float('nan')
        if flag:  # True --> give x calculate y
            if self.func_weight[0] == 0 and self.func_weight[1] != 0:
                return float(self.func_weight[2]) / self.func_weight[1]
            elif self.func_weight[0] != 0 and self.func_weight[1] == 0:
                return float('nan')
            else:
                return -1.0 * (self.func_weight[0] * val + self.func_weight[2]) / self.func_weight[1]
        else:
            if self.func_weight[0] != 0 and self.func_weight[1] == 0:
                return float(self.func_weight[2]) / self.func_weight[0]
            elif self.func_weight[0] == 0 and self.func_weight[1] != 0:
                return float('nan')
            else:
                return -1.0 * (self.func_weight[1] * val + self.func_weight[2]) / self.func_weight[0]

    def get_line(self, image):
        pt1 = (0, 0)
        pt2 = (0, 0)
        extreme_left = 0 - self.origin_coord[0]
        extreme_left_y = self.cul_line_func(True, extreme_left)

        extreme_top = self.origin_coord[1]
        extreme_top_x = self.cul_line_func(False, extreme_top)

        extreme_right = self.origin_coord[0]
        extreme_right_y = self.cul_line_func(True, extreme_right)

        extreme_bottom = 0 - self.origin_coord[1]
        extreme_bottom_x = self.cul_line_func(False, extreme_bottom)
        line_color = (0, 0, 255)  # BGR
        thickness = 2
        lineType = 4
        if not math.isnan(extreme_left_y) and extreme_left_y in range(extreme_bottom, extreme_top + 1):
            pt1 = (extreme_left + extreme_right, int(extreme_top - extreme_left_y))
        if not math.isnan(extreme_top_x) and extreme_top_x in range(extreme_left, extreme_right + 1):
            if pt1 == (0, 0):
                pt1 = (int(extreme_top_x + extreme_right), 0)
            else:
                pt2 = (int(extreme_top_x + extreme_right), 0)
                return cv2.line(image, pt1, pt2, line_color, thickness, lineType)
        if not math.isnan(extreme_right_y) and extreme_right_y in range(extreme_bottom, extreme_top + 1):
            if pt1 == (0, 0):
                pt1 = (extreme_right + extreme_right, int(extreme_top - extreme_right_y))
            else:
                pt2 = (extreme_right + extreme_right, int(extreme_top - extreme_right_y))
                return cv2.line(image, pt1, pt2, line_color, thickness, lineType)
        if not math.isnan(extreme_bottom_x) and extreme_bottom_x in range(extreme_left, extreme_right + 1):
            if pt1 == (0, 0):
                return image
            else:
                pt2 = (int(extreme_bottom_x + extreme_right), extreme_top - extreme_bottom)
                return cv2.line(image, pt1, pt2, line_color, thickness, lineType)

    def stop(self):
        self.Painting = False

    def begin(self):

        self.Painting = True

    def run(self):

        while self.Painting:
            self.qmut_1.lock()
            if self.queue.qsize() > 0:
                image = self.queue.get()
                image = self.get_line(image)
                # image = self.get_cover(image)
                self.detOut.emit(image)
                self.msleep(30)
                self.queue.task_done()
            self.qmut_1.unlock()
        self.qmut_1.unlock()
        print("Stop Painting!!!")


class ProductThread(QThread):
    sinOut = pyqtSignal(numpy.ndarray)

    def __init__(self, cap, parent=None):
        super(ProductThread, self).__init__(parent)
        # 设置工作状态与初始num数值
        # self.s_rtsp = "rtsp://iscas:opqwer12@192.168.100.176:554/Streaming/Channels/101"
        # self.cap = cv2.VideoCapture()
        self.cap = cap

        self.Producting = False

    def stop(self):
        self.Producting = False

    def begin(self):
        self.Producting = True

    def run(self):

        while self.Producting:
            print("Reading.....")
            print("Cap_Status: " + str(self.cap.isOpened()))
            # 获取文本
            flag, image = self.cap.read()
            if flag:
                # 发射信号
                self.sinOut.emit(image)
                self.msleep(30)
        print("Stop Reading !!!")


class ConsumeThread(QThread):
    sum_person = pyqtSignal(int)
    bbox_id = pyqtSignal(list)

    def __init__(self, qmut_1, queue, parent=None):
        super(ConsumeThread, self).__init__(parent)
        self.queue = queue
        self.Consuming = False
        self.qmut_1 = qmut_1

    def stop(self):
        self.Consuming = False

    def begin(self):
        self.Consuming = True

    def opts(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path')
        parser.add_argument('--source', type=str, default='rtsp://iscas:opqwer12@192.168.100.176:554/Streaming'
                                                          '/Channels/101', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        # class 0 is person
        parser.add_argument('--classes', nargs='+', type=int, default=range(0, 1), help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        return args

    def detect(self, opt):
        print("before detect lock")
        self.qmut_1.lock()
        print("after detect lock")
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader

        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            view_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        self.t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            if not self.Consuming:
                # dataset.stop_cap()
                raise StopIteration
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if not self.Consuming:
                    # dataset.stop_cap()
                    raise StopIteration
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    n = 0
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # 将当前帧的总人数发送给前端pyqt界面
                    self.sum_person.emit(n)
                    self.msleep(30)
                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        img_h, img_w, _ = im0.shape
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.bbox_id.emit([bbox_xyxy, identities])
                        self.msleep(30)
                        draw_boxes(im0, bbox_xyxy, identities)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                if view_img:
                    # self.detOut.emit(im0)
                    self.queue.put(im0)
                    # if self.queue.qsize() > 3:
                    self.qmut_1.unlock()
                    if self.queue.qsize() > 1:

                        self.queue.get(False)
                        self.queue.task_done()
                    else:
                        self.msleep(30)

        print('Done. (%.3fs)' % (time.time() - self.t0))

    def run(self):

        print("self.Consuming-->" + str(self.Consuming))
        if self.Consuming:
            try:
                print("Begin Detecting!!!!")
                with torch.no_grad():
                    self.detect(self.opts())
            except StopIteration:
                print("SOPT ITERATION!!!!!")
            finally:
                # print("Clear The Queue!!")
                # 不清理队列的时候反倒是不会有Bug....
                # 也不知道这个的原理是什么
                # 在这里mark一下，以后有机会了学习学习.....
                # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
                # This Bug again..
                while True:
                    print("Clear The Queue!!")
                    if not self.queue.empty():
                        self.queue.get(False)
                        # self.msleep(30)
                        self.queue.task_done()
                    else:
                        break

                print('Done. (%.3fs)' % (time.time() - self.t0))
                print("Stop!!!!!!!!!!!!")


if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())
