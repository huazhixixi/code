from PyQt5.QtWidgets import  QApplication
from GuiFrontEnd import Ui_Form
from PyQt5.QtWidgets import QWidget

class Simulator(QWidget):

    def __init__(self,parent=None):
        super(Simulator, self).__init__(parent=parent)
        self.__ui = Ui_Form()
        self.__ui.setupUi(self)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    test = Simulator()
    test.show()
    app.exec()