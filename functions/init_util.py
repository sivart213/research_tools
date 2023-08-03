# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:11:18 2023.

@author: j2cle

Generate the list of functions and classes to use for init.
Latest file structure to use for target:
    defect_code:
        defect_code.equations
        defect_code.functions
        defect_code.impedance_analysis
"""
if __name__ == "__main__":
    import inspect as ins
    import defect_code.equations as target
    ##



    modules = ins.getmembers(target, ins.ismodule)
    res = {}
    for targ in modules:
        ftmp = ins.getmembers(targ[1], ins.isfunction)
        ctmp = ins.getmembers(targ[1], ins.isclass)
        tmp = ftmp + ctmp
        res[targ[0]] = sorted([t[0] for t in tmp if targ[1].__name__ == t[1].__module__], key=str.lower)
        print(targ[0])
        print(f"--------")
        print(*res[targ[0]], sep = ",\n")
        print(res[targ[0]])
        print("")
