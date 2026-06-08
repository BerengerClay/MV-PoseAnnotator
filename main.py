import sys
import argparse
from PyQt6.QtWidgets import QApplication
from src.mainwindow import TrampolineAnnotator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-View Trampoline Jumper Annotator")
    parser.add_argument("paths", nargs="*", default=[], help="Path to sequence directory or 8 camera folders")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TrampolineAnnotator(paths=args.paths)
    window.show()
    sys.exit(app.exec())
