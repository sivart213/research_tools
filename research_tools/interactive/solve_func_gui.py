# -*- coding: utf-8 -*-
"""
Created on February 27, 2025

@author: JClenney

General function file
"""
# import os
import re
import sys
# import inspect
# import importlib
# from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    # QTextEdit,
    QFormLayout,
    QDialog,
    # QDialogButtonBox,
    QFrame,
    QMainWindow,
)

# from PyQt5.QtGui import QTextCursor

from .gui_tools import ModuleLoader, OptionsDialog, CollapsibleText, sci_note, safe_eval
from .solve_for_gui import SolveForGUI


class FunctionGUI(QMainWindow):
    """
    GUI for calculating functions.
    """

    def __init__(self, sources):
        super().__init__()
        self.module_loader = ModuleLoader(sources)
        self.options = {"sci_notation": True, "precision": 2}
        self.arg_widgets = {}
        self.init_ui()

    def init_ui(self):
        """
        Create the GUI layout.
        """
        self.setWindowTitle("Function GUI")
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
        submodule_label = QLabel("Select Submodule:")

        function_label = QLabel("Select Function:")

        args_label = QLabel("Arguments:")

        # Create ComboBoxes
        self.submodule_combobox = QComboBox()
        self.submodule_combobox.addItems(
            self.module_loader.module_hierarchy.keys()
        )
        self.submodule_combobox.currentIndexChanged.connect(
            self.on_submodule_change
        )

        self.function_combobox = QComboBox()
        self.function_combobox.currentIndexChanged.connect(
            self.on_function_change
        )

        # Create CollapsibleTexts
        self.doc_text = CollapsibleText("", central_widget)

        self.result_text = CollapsibleText("", central_widget, False)

        # Create Buttons
        options_button = QPushButton("Options")
        options_button.setFixedWidth(100)
        options_button.clicked.connect(self.show_options_dialog)

        solve_button = QPushButton("Solve for Arg")
        solve_button.setFixedWidth(100)
        # solve_button.setEnabled(False)
        solve_button.clicked.connect(self.show_solve_for_dialog)

        clear_history_button = QPushButton("Clear History")
        clear_history_button.setFixedWidth(100)
        clear_history_button.clicked.connect(self.result_text.clear)

        calculate_button = QPushButton("Calculate")
        calculate_button.setFixedWidth(150)
        calculate_button.clicked.connect(self.calculate)

        # Add widgets to layout
        layout.addWidget(submodule_label)
        layout.addWidget(self.submodule_combobox)

        layout.addWidget(function_label)
        layout.addWidget(self.function_combobox)

        layout.addWidget(self.doc_text)
        args_layout.addWidget(args_label)

        args_layout.addLayout(self.args_form_layout)
        args_frame.setLayout(args_layout)
        layout.addWidget(args_frame)

        button_layout.addWidget(options_button)
        button_layout.addWidget(solve_button)
        button_layout.addWidget(clear_history_button)
        layout.addLayout(button_layout)

        layout.addWidget(separator)

        layout.addWidget(calculate_button)
        layout.addWidget(self.result_text)

        self.on_submodule_change(0)

    def on_submodule_change(self, _):
        """
        Update the function list when the submodule changes.
        """
        submodule_name = self.submodule_combobox.currentText()
        self.function_combobox.clear()
        functions = self.module_loader.module_hierarchy[submodule_name]
        self.function_combobox.addItems(functions.keys())
        self.on_function_change(0)

    def on_function_change(self, _):
        """
        Update the docstring and arguments when the function changes.
        """

        if self.function_combobox.count() == 0:
            return
        function_key = self.function_combobox.currentText()
        submodule_name = self.submodule_combobox.currentText()
        self.doc_text.setText(
            self.module_loader.get_func_docstring(submodule_name, function_key)
        )

        params = self.module_loader.get_func_params(
            submodule_name, function_key
        )
        while self.args_form_layout.count():
            item = self.args_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.arg_widgets = {}
        for name, default, anno in params:
            label = QLabel(name)
            line_edit = QLineEdit()
            if anno:
                line_edit.setToolTip(str(anno))
            if default is not None:
                line_edit.setText(str(default))
            self.args_form_layout.addRow(label, line_edit)
            self.arg_widgets[name] = line_edit

    def show_options_dialog(self):
        """
        Method to show the options dialog.
        """
        dialog = OptionsDialog(self.options, self)
        if dialog.exec_() == QDialog.Accepted:
            self.options = dialog.getOptions()

    def show_solve_for_dialog(self):
        """
        Method to show the solve for dialog.
        """
        function_key = self.function_combobox.currentText()
        submodule_name = self.submodule_combobox.currentText()
        function = self.module_loader.module_hierarchy[submodule_name][
            function_key
        ]
        self.solve_for_window = SolveForGUI(function)


    def calculate(self):
        """
        Perform the calculation and display the result.
        """
        function_key = self.function_combobox.currentText()
        submodule_name = self.submodule_combobox.currentText()
        function = self.module_loader.module_hierarchy[submodule_name][
            function_key
        ]
        args = {}
        for name, widget in self.arg_widgets.items():
            args[name] = safe_eval(widget.text())
        try:
            result = function(**args)

            if self.options["sci_notation"]:
                if isinstance(result, (list, tuple)):
                    result = [
                        sci_note(res, self.options["precision"])
                        for res in result
                    ]
                else:
                    result = sci_note(result, self.options["precision"])
            result = self.format_result_text(
                function.__name__,
                args,
                result,
                self.module_loader.get_func_docstring(
                    submodule_name, function_key
                ),
            )
            self.result_text.insert_text(str(result))
        except Exception as e:
            self.result_text.insert_text(f"Error: {e}")

    def parse_docstring(self, docstring, expected_names=1):
        """
        Helper function to parse the docstring for the expected names and units.
        """
        # get the string section that contains the names and units of the results
        sub_string = re.search(
            r"(?i:returns?|results?)\s*:\s*(?:\w+\s*:\s*)?\s*(.*?)\s*\.?\s*(?=\n\s*\n|$)",
            docstring,
        )
        if not sub_string:
            return [{"name": "Result", "unit": ""}]

        # get the name/unit pairs.  This is what should match the expected names value
        matches = re.findall(r"(?:\w\s*)+(?:\([^)]+\))?", sub_string[1])
        if not matches:
            return [{"name": "Result", "unit": ""}]
        elif len(matches) != expected_names == 1:
            matches = [" ".join(matches)]
        elif len(matches) != expected_names > 1:
            if expected_names == len(
                sub_matches := re.split(r"[,;]", sub_string[1])
            ):
                matches = sub_matches
            elif (len(matches) % expected_names) == 0:
                seg_len = len(matches) // expected_names
                matches = [
                    " ".join(matches[i : i + seg_len])
                    for i in range(0, len(matches), seg_len)
                ]
            elif (len(sub_matches) % expected_names) == 0:
                seg_len = len(sub_matches) // expected_names
                matches = [
                    " ".join(sub_matches[i : i + seg_len])
                    for i in range(0, len(sub_matches), seg_len)
                ]
            else:
                return [{"name": "(" + sub_string[1] + ")", "unit": ""}]
        results = []
        for grp in matches:
            # Evaluate each name/unit pairs
            names = re.findall(r"((?:\b\w+\b)|\((?:.*?)\))", grp)
            unit = (
                " " + names.pop() if names[-1].startswith(("(", "[")) else ""
            )

            if len(name := " ".join([n for n in names])) > 16:
                name = ""
                for nm in names:
                    nm = nm.strip().title()
                    if len(nm) <= 3:
                        name += nm
                    elif nm[3] not in "aeiou":  # 4th not vowel
                        name += nm[:4]
                    elif nm[2] not in "aeiou":  # 3rd not vowel
                        name += nm[:3]
                    else:
                        name += nm[: min(5, len(nm))]
            elif " " in name and not name.istitle():
                name = name.title()
            results.append({"name": name, "unit": unit})
        if not results:
            return [{"name": "Result", "unit": ""}]

        return results

    def format_result_text(self, function_name, args, result, docstring):
        """
        Helper function to format the result text.
        """
        expected_len = len(result) if isinstance(result, (list, tuple)) else 1
        return_info = self.parse_docstring(docstring, expected_len)
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        if isinstance(result, (list, tuple)):
            if len(return_info) == len(result):
                result_str = ", ".join(
                    f"{info['name']} = {res}{info['unit']}"
                    for info, res in zip(return_info, result)
                )
            elif len(return_info) == 1:
                result_name = return_info[0].get("name", "res")
                result_unit = return_info[0].get("unit", "")
                result_str = ", ".join(
                    f"{result_name + str(n+1)} = {res}{return_info[0]['unit']}"
                    for n, res in enumerate(result)
                )
            else:
                result_str = "Result" + ", ".join(result)
        else:
            result_name = return_info[0].get("name", "result")
            result_unit = return_info[0].get("unit", "")
            result_str = f"{result_name} = {result} {result_unit}"
        return f"{function_name}({args_str})\n{result_str}"


if __name__ == "__main__":
    app = QApplication(sys.argv)

    default_sources = [
        "research_tools.equations",
    ]
    gui = FunctionGUI(default_sources)
    gui.show()

    sys.exit(app.exec_())
