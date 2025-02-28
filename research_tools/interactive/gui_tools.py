# -*- coding: utf-8 -*-
"""
Created on February 27, 2025

@author: JClenney

General function file
"""
import os
import re
import sys
import inspect
import importlib
from pathlib import Path

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
    QTextEdit,
    QFormLayout,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QMainWindow,
)

from PyQt5.QtGui import QTextCursor

eval_vars = {
    "np": np,
    "inf": np.inf,
    "nan": np.nan,
    "pi": np.pi,
    "e": np.e,
    "ln": np.log,
    "log": np.log,
    "log10": np.log10,
    "true": True,
    "false": False,
    "none": None,
}


def safe_eval(value):
    """
    Safely evaluate a string expression.
    """
    if not isinstance(value, str):
        return value
    try:
        return eval(value, {}, eval_vars)
    except (SyntaxError, TypeError, ValueError, NameError):
        if re.match(r"^[\[\(\{].*[\]\)\}]$", value):
            parts = re.findall(r"[^,\s]+", value[1:-1])
            if value.startswith("["):
                return [safe_eval(part) for part in parts]
            elif value.startswith("("):
                return tuple(safe_eval(part) for part in parts)
            elif value.startswith("{"):
                if ":" in parts[0]:
                    return {
                        safe_eval(k): safe_eval(v)
                        for k, v in (part.split(":") for part in parts)
                    }
                else:
                    return set(safe_eval(part) for part in parts)
        elif value.strip() != value:
            return safe_eval(value.strip())
        else:
            return value


def sci_note(num, prec=2):
    """
    Convert a number to scientific notation.
    """
    if not isinstance(num, (float, int)) or np.isnan(num) or num == np.inf:
        return str(num)
    fmt = "{:.%dE}" % int(prec)
    return fmt.format(num)


# # Create separators
# def create_separator(frame=None):
#     separator = QFrame(frame)
#     separator.setFrameShape(QFrame.HLine)
#     separator.setFrameShadow(QFrame.Sunken)
#     return separator


class OptionsDialog(QDialog):
    """
    Dialog configured to set options.
    """

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.options = options
        layout = QFormLayout()
        self.edits = {}
        for key, value in self.options.items():
            self.edits[key] = QLineEdit(str(value))
            layout.addRow(key.replace("_", " ").title(), self.edits[key])
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addRow(self.buttonBox)
        self.setLayout(layout)

    def getOptions(self):
        """
        Retrieve the options from the dialog entries.
        """
        for key, edit in self.edits.items():
            self.options[key] = safe_eval(edit.text())
        return self.options


class ModuleLoader:
    """
    Load functions from a module or modules.
    """

    def __init__(self, sources, categorize_names=True):
        self.sources = self.convert_sources(sources)
        self.module_hierarchy = self.load_functions(
            categorize_names=categorize_names
        )

    def convert_sources(self, sources):
        """
        Standardize the sources input.
        """

        def check_source(in_source):
            clean_source = {}
            for name, val in in_source.items():
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\.]*$", name):
                    raise ValueError(f"Invalid module name: {name}")
                if name.endswith(".py"):
                    name.replace(".py", "")
                if (spec := importlib.util.find_spec(name)) is not None:
                    if (
                        in_paths := spec.submodule_search_locations
                    ) is not None:
                        sub_source_names = [
                            ".".join(
                                p.relative_to(Path(in_paths[0]))
                                .with_suffix("")
                                .parts
                            )
                            for p in Path(in_paths[0]).glob("**/[!_]*.py")
                        ]
                        clean_source.update(
                            {f"{name}.{k}": val for k in sub_source_names}
                        )
                    elif name.endswith("__init__") and not val:
                        module = importlib.import_module(name)
                        for fname, func in inspect.getmembers(module):
                            if not inspect.isfunction(func) or not (
                                submodule := inspect.getmodule(func)
                            ):
                                continue
                            source_name = submodule.__name__
                            if (
                                source_name in clean_source
                                and clean_source[source_name]
                            ):
                                clean_source[source_name].append(fname)
                            else:
                                clean_source[source_name] = [fname]

                    else:
                        clean_source[name] = val
            return clean_source

        if sources is None:
            res = check_source({os.getcwd(): None})
        elif isinstance(sources, str):
            res = check_source({sources: None})
        elif isinstance(sources, (list, tuple)):
            res = check_source({source: None for source in sources})
        elif isinstance(sources, dict):
            res = check_source(sources)
        else:
            raise ValueError("Invalid type for sources")
        return res

    def load_functions(self, sources=None, categorize_names=True):
        """
        Load functions from the specified sources.
        """
        sources = sources or self.sources

        module_hierarchy = {}
        for module_name, functions in sources.items():
            base_module = importlib.import_module(module_name)
            current_category = None
            submodule_path = inspect.getfile(base_module)
            with open(submodule_path, "r", encoding="utf-8") as file:  #
                for line in file:
                    if line.startswith("# %%"):
                        current_category = line[4:].strip()
                    elif line.startswith("def "):
                        function_name = line.split("(")[0][4:].strip()
                        if functions and function_name not in functions:
                            continue
                        function = getattr(base_module, function_name, None)
                        if function:
                            source_name = inspect.getmodule(function).__name__
                            if source_name not in module_hierarchy:
                                module_hierarchy[source_name] = {}
                            if categorize_names and current_category:
                                key = f"{current_category} - {function_name}"
                            else:
                                key = function_name
                            module_hierarchy[source_name][key] = function
        return module_hierarchy

    def get_func_docstring(self, module_name, function_key):
        """
        Helper function to get the docstring for a function.
        """
        function = self.module_hierarchy[module_name][function_key]
        docstring = inspect.getdoc(function)
        if not docstring:
            if "-" in function_key:
                category, func_name = function_key.split("-")[:2]
                docstring = f"Function to calculate {func_name.strip()} in the {category.strip()} category as found in {module_name}."
            else:
                docstring = f"Function to calculate {function_key} as found in {module_name}."
        return docstring

    def get_func_params(self, module_name, function_key):
        """
        Helper function to get the parameters for a function.
        """
        function = self.module_hierarchy[module_name][function_key]
        sig = inspect.signature(function)
        for param in sig.parameters.values():
            default = (
                param.default
                if param.default != inspect.Parameter.empty
                else None
            )
            annotation = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            )
            yield param.name, default, annotation


class CollapsibleText(QTextEdit):
    """
    Multiline text box that can be collapsed.
    """

    def __init__(self, content="", parent=None, collapsible=True):
        super().__init__(content, parent)
        self.setReadOnly(True)
        self._collapsible = bool(collapsible)
        if self._collapsible:
            self.setFixedHeight(40)  # Adjust height as needed
            self.collapsed = True

    def mouseDoubleClickEvent(self, event):
        """
        Redefine the mouseDoubleClickEvent to toggle the content.
        """
        self.toggle_content()
        super().mouseDoubleClickEvent(event)

    def toggle_content(self):
        """
        Changes the height of the text box to show or hide the content.
        """

        if not self._collapsible:
            return
        if self.collapsed:
            self.setMaximumHeight(16777215)
            document_height = self.document().size().height()
            new_height = int(min(document_height, self.width() / 2))
            height_diff = new_height - self.height()
            self.resize(self.width(), new_height)
        else:

            height_diff = 40 - self.height()
            self.setFixedHeight(40)

        self.collapsed = not self.collapsed

        # Adjust the parent window's height
        parent = self.parentWidget()
        if parent:
            parent.resize(parent.width(), parent.height() + height_diff)

    def insert_text(self, content):
        """
        Automatically insert text and scroll to the top.
        """
        cursor = QTextCursor(self.document())
        cursor.setPosition(0)
        self.setTextCursor(cursor)
        self.insertPlainText(content + "\n\n")

    def collapsible(self, state=None):
        """
        Method to get or set the collapsible state.
        """
        if state is not None:
            self._collapsible = bool(state)
        return self._collapsible

