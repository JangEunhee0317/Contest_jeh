import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *

form_class = uic.loadUiType("Contest_jeh_Dialog.ui")[0]
class dialog(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('timer_icon_153935.png'))

        
        # 상단바없애기
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        # 윈도우창위치조정
        self.move( 1160, 200 )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = dialog()
    myWindow.show()