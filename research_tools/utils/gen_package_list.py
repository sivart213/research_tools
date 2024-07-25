# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:01:42 2023

@author: j2cle
"""

import ast
from pathlib import Path
from research_tools.functions import find_path, find_files

import subprocess
import pandas as pd
import json
import importlib_metadata
import yaml



class node_iter(ast.NodeVisitor):
    def __init__(self, exclude=None):
        self.modules = set()
        self.exclude = exclude

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, value):
        if value is None:
            self._exclude = []
        elif isinstance(value, str):
            self._exclude = [value]
        else:
            self._exclude = list(value)

    def visit_Import(self, node):
        for name in node.names:
            res = name.name.split(".")[0]
            if res not in self.exclude:
                self.modules.add(res)

    def visit_ImportFrom(self, node):
        # if node.module is missing it's a "from . import ..." statement
        # if level > 0 it's a "from .submodule import ..." statement
        if node.module is not None and node.level == 0:
            res = node.module.split(".")[0]
            if res not in self.exclude:
                self.modules.add(res)


class dir_iter(object):
    def __init__(self, default=None, exclude=None):
        if default is None:
            default = ["pip", "python", "spyder"]
        self.modules = set(default)
        self.exclude = exclude
        self.tree = dict()

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, value):
        if value is None:
            self._exclude = []
        elif isinstance(value, str):
            self._exclude = [value]
        else:
            self._exclude = list(value)

    def walk(self, path):
        if isinstance(path, (str, Path)):
            node_objs = node_iter(self.exclude)
            with open(str(path.absolute())) as f:
                node_objs.visit(ast.parse(f.read()))
            self.tree[path.stem] = node_objs.modules
            self.modules.update(node_objs.modules)
        elif hasattr(path, "__iter__"):
            return [self.walk(p) for p in path]

    def packages(self, env=None):
        instr = ["conda", "list", "--json"]
        if env:
            instr = instr + ["-n", env]

        df = pd.json_normalize(
            json.loads(
                subprocess.run(instr, stdout=subprocess.PIPE).stdout.decode(
                    "utf-8"
                )
            )
        )

        dists = importlib_metadata.packages_distributions()
        pkgs = [
            dists[m][0]
            if m in self.modules.difference(df["name"]) and m in dists.keys()
            else m
            for m in self.modules
        ]

        return df.loc[df["name"].isin(pkgs)].reset_index(drop=True)

def gen_env_list(files, env_nm = "env", exclude=None, out_path=None, version=False, python_version=True):

    mod_res = dir_iter(exclude=exclude)
    mod_res.walk(files)
    mods = mod_res.modules

    pkgs = mod_res.packages("my_env")
    
    
    pkgs_nonpip = pkgs.loc[pkgs["channel"] != "pypi"].reset_index(drop=True)
    pkgs_nonpip["name_version"] = pkgs_nonpip["name"] + "=" + pkgs_nonpip["version"]
    
    pkgs_pip = pkgs.loc[pkgs["channel"] == "pypi"].reset_index(drop=True)
    pkgs_pip["name_version"] = pkgs_pip["name"] + "==" + pkgs_pip["version"]

    chan = list(set(pkgs["channel"]).difference(set(["pkgs/main", "pypi"])))
    chan.append("defaults")

    dep_nm = "name"
    if version:
        dep_nm = "name_version"  
    if not version and python_version:
        depd = [
            d
            if d != "python"
            else ".".join(pkgs_nonpip.loc[n, "name_version"].split(".")[:2])
            for n, d in enumerate(pkgs_nonpip[dep_nm])
        ]
    else:
        depd = list(pkgs_nonpip[dep_nm])
    
    if pkgs.isin(["pypi"]).any().any():
        depd.append(dict(pip=list(pkgs_pip[dep_nm])))

    if out_path is not None and Path(out_path).exists():
        with open(Path(out_path)/f"{env_nm}.yml", "w") as file:
            yaml.dump(
                dict(
                    name=env_nm,
                    channels=chan,
                    dependencies=depd,
                ),
                file,
                sort_keys=False,
                default_flow_style=False,
            )
        return
    else:
        return depd
    
target = ["research_tools"]
# target = ["impedance_analysis", "ion_migration", "iv_analysis"]

# files = find_files(find_path(target, base=Path.cwd()), patterns=r"[^_][.]py$")
files = [
    find_files(find_path(t, base=Path.cwd()), patterns=r"[^_][.]py$") for t in target
]
files = [
    item for row in files for item in row if "archive" not in str(item).lower()
]
exclude = [f.stem for f in files]

gen_env_list(files, "requirements", exclude, Path.home(), True, False)


