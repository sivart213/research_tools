# -*- coding: utf-8 -*-
"""
Created on February 27, 2025

@author: JClenney

General function file
"""
# import os
# import re
import sys
import inspect
# import importlib
# from pathlib import Path

# import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    # QComboBox,
    # QTextEdit,
    QFormLayout,
    QDialog,
    # QDialogButtonBox,
    QFrame,
    QMainWindow,
)


from .gui_tools import ModuleLoader, OptionsDialog, CollapsibleText, sci_note, safe_eval

from ..functions import solve_for_variable

class SolveForGUI(QMainWindow):
    """
    GUI for solving functions using solve_for_variable.
    """

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.options = {"sci_notation": True, "precision": 2}
        self.arg_widgets = {}
        self.init_ui()

    def init_ui(self):
        """
        Create the GUI layout.
        """
        self.setWindowTitle("Solve For GUI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # Args layout with label and frame
        args_frame = QFrame()
        args_frame.setFrameShape(QFrame.StyledPanel)
        args_frame.setFrameShadow(QFrame.Raised)
        args_layout = QVBoxLayout()

        self.args_form_layout = QFormLayout()

        button_layout = QHBoxLayout()
        # Create Labels
        args_label = QLabel("Arguments:")


        solve_for_label = QLabel("Solve For:")
        dep_var_label = QLabel("Dependent Variable:")
        form_label = QLabel("Form:")

        # Create LineEdits
        self.solve_for_edit = QLineEdit()
        self.dep_var_edit = QLineEdit()
        self.form_edit = QLineEdit()

        # Create CollapsibleTexts
        self.function_doc_text = CollapsibleText(inspect.getdoc(self.function), central_widget)
        self.solve_for_doc_text = CollapsibleText(inspect.getdoc(solve_for_variable), central_widget)
        self.result_text = CollapsibleText("", central_widget, False)

        # Create Buttons
        options_button = QPushButton("Options")
        options_button.setFixedWidth(100)
        options_button.clicked.connect(self.show_options_dialog)

        clear_history_button = QPushButton("Clear History")
        clear_history_button.setFixedWidth(100)
        clear_history_button.clicked.connect(self.result_text.clear)

        calculate_button = QPushButton("Calculate")
        calculate_button.setFixedWidth(150)
        calculate_button.clicked.connect(self.calculate)

        # Add widgets to layout
        layout.addWidget(self.solve_for_doc_text)
        layout.addWidget(self.function_doc_text)
        args_layout.addWidget(args_label)

        args_layout.addLayout(self.args_form_layout)
        args_frame.setLayout(args_layout)
        layout.addWidget(args_frame)

        layout.addWidget(solve_for_label)
        layout.addWidget(self.solve_for_edit)

        layout.addWidget(dep_var_label)
        layout.addWidget(self.dep_var_edit)

        layout.addWidget(form_label)
        layout.addWidget(self.form_edit)

        button_layout.addWidget(options_button)
        button_layout.addWidget(clear_history_button)
        layout.addLayout(button_layout)

        layout.addWidget(separator)

        layout.addWidget(calculate_button)
        layout.addWidget(self.result_text)

        self.populate_args()
        self.show()

    def populate_args(self):
        """
        Populate the argument fields based on the function signature.
        """
        params = inspect.signature(self.function).parameters
        for name, param in params.items():
            label = QLabel(name)
            line_edit = QLineEdit()
            if param.annotation != inspect.Parameter.empty:
                line_edit.setToolTip(str(param.annotation))
            if param.default != inspect.Parameter.empty:
                line_edit.setText(str(param.default))
            self.args_form_layout.addRow(label, line_edit)
            self.arg_widgets[name] = line_edit

    def show_options_dialog(self):
        """
        Method to show the options dialog.
        """
        dialog = OptionsDialog(self.options, self)
        if dialog.exec_() == QDialog.Accepted:
            self.options = dialog.getOptions()

    def calculate(self):
        """
        Perform the calculation and display the result.
        """
        args = {}
        for name, widget in self.arg_widgets.items():
            args[name] = safe_eval(widget.text())
        solve_for = self.solve_for_edit.text()
        dep_var = self.dep_var_edit.text()
        form = self.form_edit.text()
        try:
            result = solve_for_variable(self.function, solve_for, dep_var, form, **args)

            if self.options["sci_notation"]:
                if isinstance(result, (list, tuple)):
                    result = [
                        sci_note(res, self.options["precision"])
                        for res in result
                    ]
                else:
                    result = sci_note(result, self.options["precision"])
            self.result_text.insert_text(str(result))
        except Exception as e:
            self.result_text.insert_text(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example function to be used with SolveForGUI
    def example_function(x, y, z=1):
        return x + y + z

    gui = SolveForGUI(example_function)
    gui.show()

    sys.exit(app.exec_())