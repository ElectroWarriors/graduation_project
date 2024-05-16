import os
import sys
# import numpy as np
from matplotlib import pyplot as plt

import Optimizer
import user

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from PyQt5.QtCore import QThread, pyqtSignal, QProcess
class Worker(QThread):
    timeout = pyqtSignal()
    def __init__(self, 
                 SimFilePath, SimFileName, LTspiceExec, Model, 
                 sim_sel, lb, ub, circuitsimulation, startpoint, 
                 Avomin, GBWmin, PMmin):
        super().__init__()
        self.iter = []
        self.hisy = []
        self.optimizer = Optimizer.optimizer(SimFilePath=SimFilePath, SimFileName=SimFileName, LTspiceExec=LTspiceExec, Model=Model, 
                                             sim_sel=sim_sel, lb=lb, ub=ub, circuitsimulation=circuitsimulation, startpoint=startpoint, 
                                             Avomin=Avomin, GBWmin=GBWmin, PMmin=PMmin)
    def run(self):
        self.optimizer.optimize()
        self.iter = self.optimizer.history_iter
        self.hisy = self.optimizer.history_min
        print(self.hisy)
        # plt.plot(self.iter, self.hisy)
        # plt.show()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(500, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.SimMethod = QComboBox(self.centralwidget)
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.addItem("")
        self.SimMethod.setObjectName(u"SimMethod")
        self.SimMethod.setGeometry(QRect(160, 50, 67, 22))

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(80, 120, 111, 21))

        self.Library = QComboBox(self.centralwidget)
        self.Library.addItem("")
        self.Library.addItem("")
        self.Library.addItem("")

        self.Library.setObjectName(u"Library")
        self.Library.setGeometry(QRect(160, 120, 90, 22))

        self.label_0 = QLabel(self.centralwidget)
        self.label_0.setObjectName(u"label_0")
        self.label_0.setGeometry(QRect(80, 50, 81, 21))

        self.UseCircuitSimulation = QRadioButton(self.centralwidget)
        self.UseCircuitSimulation.setObjectName(u"UseCircuitSimulation")
        self.UseCircuitSimulation.setGeometry(QRect(240, 50, 161, 21))
        self.UseCircuitSimulation.setCheckable(True)

        self.SimFilePath_Name = QLineEdit(self.centralwidget)
        self.SimFilePath_Name.setObjectName(u"SimFilePath_Name")
        self.SimFilePath_Name.setGeometry(QRect(200, 80, 171, 20))

        self.label_1 = QLabel(self.centralwidget)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(80, 80, 111, 21))

        self.OpenSimFile = QPushButton(self.centralwidget)
        self.OpenSimFile.setObjectName(u"OpenSimFile")
        self.OpenSimFile.setGeometry(QRect(380, 80, 41, 23))
        self.OpenSimFile.clicked.connect(self.OpenSimFileCallback)

        self.Run = QPushButton(self.centralwidget)
        self.Run.setObjectName(u"Run")
        self.Run.setGeometry(QRect(100,300, 80, 30))
        self.Run.clicked.connect(self.RunCallback)

        self.Stop = QPushButton(self.centralwidget)
        self.Stop.setObjectName(u"Stop")
        self.Stop.setGeometry(QRect(320, 300, 80, 30))
        self.Stop.clicked.connect(self.StopCallback)

        self.timer0 = QTimer()
        self.timer0.timeout.connect(self.Timer0)
        self.timer_switch_flag = False
        # self.graphicsView = QGraphicsView(self.centralwidget)
        # self.graphicsView.setObjectName(u"graphicsView")
        # self.graphicsView.setGeometry(QRect(90, 160, 321, 261))

        self.Report = QTextEdit(self.centralwidget)
        self.Report.setObjectName(u"Report")
        self.Report.setGeometry(QRect(80, 150, 340, 130))

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 796, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"模拟电路参数优化工具", None))
        self.SimMethod.setItemText(0, QCoreApplication.translate("MainWindow", u"SA", None))
        self.SimMethod.setItemText(1, QCoreApplication.translate("MainWindow", u"GA", None))
        self.SimMethod.setItemText(2, QCoreApplication.translate("MainWindow", u"DE", None))
        self.SimMethod.setItemText(3, QCoreApplication.translate("MainWindow", u"PSO", None))
        self.SimMethod.setItemText(4, QCoreApplication.translate("MainWindow", u"BO", None))
        self.SimMethod.setItemText(5, QCoreApplication.translate("MainWindow", u"PSOGA", None))
        self.SimMethod.setItemText(6, QCoreApplication.translate("MainWindow", u"GANG", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u5de5\u827a\u5e93", None))
        self.Library.setItemText(0, QCoreApplication.translate("MainWindow", u"smic18", None))
        self.Library.setItemText(1, QCoreApplication.translate("MainWindow", u"tsmc18", None))
        self.Library.setItemText(2, QCoreApplication.translate("MainWindow", u"smic13", None))

        self.label_0.setText(QCoreApplication.translate("MainWindow", u"\u4f18\u5316\u65b9\u6cd5", None))
        self.UseCircuitSimulation.setText(QCoreApplication.translate("MainWindow", u"\u91c7\u7528\u7535\u8def\u4eff\u771f", None))

        self.UseCircuitSimulation.setShortcut("")

        self.label_1.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u4eff\u771f\u6587\u4ef6", None))
        self.OpenSimFile.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00", None))
        self.Run.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.Stop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))


    def OpenSimFileCallback(self):
        #设置文件扩展名过滤,注意用双分号间隔:"All Files (*);;Files (*.asc)"
        SimFilePath_Name, filetype = QFileDialog.getOpenFileName(None, "选取文件", None, "Files (*.asc);;All Files (*)")  
        self.SimFilePath_Name.setText(SimFilePath_Name)

    def RunCallback(self):
        LTspiceExec = user.LTspiceExec # "D:\\DATA\\Software\\LTspice\\LTspice.exe"
        sim_sel = self.SimMethod.currentText()
        library = self.Library.currentText()
        circuitsimulation = False
        if(self.UseCircuitSimulation.isChecked()):
            circuitsimulation = True

        SimFilePath_Name = self.SimFilePath_Name.text()
        SimFilePath, SimFileName = os.path.split(SimFilePath_Name)
        SimFileName, asc = os.path.splitext(SimFileName)
        print(SimFilePath, SimFileName)

        self.timer0.start(1000)
        self.timer_switch_flag = True

        self.worker = Worker(SimFilePath=SimFilePath+"/", SimFileName=SimFileName, LTspiceExec=LTspiceExec, Model=library, 
                        sim_sel=sim_sel, lb=user.lower_bound, ub=user.upper_bound, circuitsimulation=circuitsimulation, startpoint=True, 
                        Avomin=90, GBWmin=20E6, PMmin=60)
        self.worker.start()

            

    def StopCallback(self):
        sys.exit()

    def Timer0(self):
        if(len(self.worker.iter) != 0):
            self.timer0.stop()
            self.Report.setText("x_star: " + str(self.worker.optimizer.best_x) + "\n" + \
                                "y_star: " + str(self.worker.optimizer.best_y))
            plt.plot(self.worker.iter, self.worker.hisy)
            plt.title("simulation result")
            plt.show()
            


if __name__ == "__main__":
    print(sys.argv)
    path, _ = os.path.split(sys.argv[0])
    os.chdir(path)

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setStyleSheet("#MainWindow{border-image:url(background3.png)}")
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
    pass