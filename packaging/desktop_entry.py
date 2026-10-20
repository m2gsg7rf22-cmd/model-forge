"""Frozen desktop entry with an explicit acceptance-test mode."""

import sys
from modelforge.gui import MainWindow
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

app = QApplication(sys.argv)
window = MainWindow()
window.show()
if "--smoke-test" in sys.argv:
    QTimer.singleShot(1500, app.quit)
sys.exit(app.exec())
