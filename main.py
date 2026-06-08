import sys
import argparse
from PyQt6.QtWidgets import QApplication
from src.mainwindow import TrampolineAnnotator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-View Trampoline Jumper Annotator")
    parser.add_argument("sequence_dir", nargs="?", default=None, help="Path to the sequence directory (optional)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TrampolineAnnotator(sequence_dir=args.sequence_dir)
    window.show()
    sys.exit(app.exec())
