#! /usr/bin/env python
# -*-coding:utf-8 -*

__author__ = "Pascal Lefevre"
__email__ = "plefevre561@gmail.com"

"""This Python 2 module contains tools for manipulating Linux paths.
EPath is the main class and proposes lots features.
"""


import sys
import os
from shutil import copyfile
import pathlib
import cv2
import image

def replace_dir(path_obj, newdir):
    name = os.path.join(newdir, path_obj.name)
    return name

def replace_suffix(path_obj, suffix):
    """replaces last suffix of path"""
    if suffix.startswith('.'):
        basename = "".join([path_obj.stem, suffix])
    else:
        basename = "".join([path_obj.stem, '.', suffix])
    return os.path.join(str(path_obj.parent), basename)

def add_before_stem(path_obj, ssuffix):
    """add suffix (ssufix) before stem"""
    basename = "".join([ssuffix, path_obj.stem, path_obj.suffix])
    name = os.path.join(str(path_obj.parent), basename)
    return name

def add_after_stem(path_obj, ssuffix):
    """add suffix (ssufix) after stem"""
    basename = "".join([path_obj.stem, ssuffix, path_obj.suffix])
    name = os.path.join(str(path_obj.parent), basename)
    return name


class EPath:
    """Enhanced Path class with custom features
    default behavior of functions : return str objects
    odd case : -- import pathtools as pt
               -- ep = pt.EPath("/tmp/ttt/file.a.b.c")
               -- ep.add_after_stem("extra_param_")
               >> '/tmp/ttt/file.a.bextra_param_.c'
    does not support urls yet"""
    def __init__(self, obj):
        if isinstance(obj, str):
            if obj.endswith('/'):
                self.path_str = obj[:-1]
            else:
                self.path_str = obj
            self.path_obj = pathlib.Path(self.path_str)
        elif isinstance(obj, pathlib.Path):
            self.path_obj = obj
            self.path_str = str(obj)
        elif isinstance(obj, EPath):
            self.path_obj = obj.path_obj
            self.path_str = obj.path_str
        else:
            raise ValueError("not a str, not a pathlib.Path obj, "
                             "not a WPath, "
                             "not a EPath !")

    def parent(self):
        return EPath(str(self.path_obj.parent))

    def stem(self):
        return EPath(self.path_obj.stem)

    def stem_noparam(self):
        return EPath(self.stem().string().split('_')[0])

    def basename(self):
        return EPath(self.path_obj.name)

    def suffix(self):
        return EPath(self.path_obj.suffix)

    def has_suffix(self):
        if len(self.path_obj.suffix) == 0:
            return False
        else:
            return True

    def add_suffix(self, suffix):
        if suffix.startswith("."):
            suffix = suffix[1:]
        p = ".".join([self.path_str, suffix])
        return EPath(p)
    sc_as = add_suffix
    def exists(self):
        return self.path_obj.exists()

    def is_dir(self):
        return self.path_obj.is_dir()

    def is_file(self):
        return self.path_obj.is_file()

    def is_readable(self):
        return os.access(self.path_str, os.R_OK)

    def is_writable(self):
        return os.access(self.path_str, os.W_OK)

    def is_executable(self):
        return os.access(self.path_str, os.X_OK)

    def mkdir(self, raiseException=False):
        """silent mkdir"""
        if not self.exists():
            self.path_obj.mkdir()
        else:
            if raiseException:
                self.path_obj.mkdir()

    def touch(self):
        """creates the file but does not erase its content if it exists"""
        self.path_obj.touch()

    def removefile(self):
        if not self.is_file() and self.exists():
            raise ValueError("This is not a file !")
        else:
            if self.exists():
                return os.remove(self.path_str)

    def string(self):
        return self.__str__()

    def replace_parents(self, new_parents, obj=True):
        """replaces all parents by np, if obj is True, returns an object"""
        path = os.path.join(str(new_parents), self.basename().string())
        return EPath(path) if obj else path
    sc_rp = replace_parents

    def replace_suffix(self, new_suffix, obj=True):
        """replaces last suffix of path"""
        if new_suffix.startswith('.'):
            basename = "".join([self.stem().string(), new_suffix])
        else:
            basename = "".join([self.stem().string(), '.', new_suffix])
        path = os.path.join(self.parent().string(), basename)
        return EPath(path) if obj else path
    sc_rs = replace_suffix

    def add_before_stem(self, ssuffix, sep='_', obj=True):
        """add suffix (ssufix) before stem"""
        basename = "".join(
            [
            ssuffix,
            sep,
            self.stem().string(),
            self.suffix().string()
            ])
        path = os.path.join(self.parent().string(), basename)
        return EPath(path) if obj else path
    sc_abs = add_before_stem

    def add_after_stem(self, ssuffix, sep='_', obj=True):
        """add suffix (ssufix) after stem"""
        basename = "".join(
            [self.stem().string(),
             sep,
             ssuffix,
             self.suffix().string()
             ])
        path = os.path.join(self.parent().string(), basename)
        return EPath(path) if obj else path
    sc_aas = add_after_stem

    def add_param(self, psuffix, sep='_', obj=True):
        """add parameters suffix after stem"""
        # if dict then name params in psuffix
        # else just put values
        if isinstance(psuffix, dict):
            keys = list(map(str, psuffix.keys()))
            values = list(map(str, psuffix.values()))
            ssuffix = "_".join("".join(item) for item in zip(keys, values))
        elif isinstance(psuffix, list) or isinstance(psuffix, tuple):
            ssuffix = "_".join(map(str, psuffix))
        else:
            raise ValueError("psuffix has not the right type")

        return self.add_after_stem(ssuffix, obj=obj)


    def join(self, extrapath, obj=True, makedir=False):
        if isinstance(extrapath, list) or isinstance(extrapath, tuple):
            r = os.path.join(self.path_str, *extrapath)
        else:
            r = os.path.join(self.path_str, str(extrapath))
        assert obj in [True, False]
        if makedir:
            if not os.path.exists(r):
                os.mkdir(r)
        return EPath(r) if obj else str(r)

    def __add__(self, extrasuffix):
        """concatenate extrasuffix with path_str stem
           eg : /tmp/file + hello --> /tmp/filehello"""
        # assert not extrasuffix.startswith("/"), "use / operator instead"
        r = self.path_str + extrasuffix
        return EPath(r)

    def __truediv__(self, extrapath):
        """simulates / like linux paths but in Python code
           returns an EPath object"""
        return self.join(extrapath)

    def __getitem__(self, item):
        """parents and basename access in a list
           it is basically indexed access to the split path by /"""
        items = self.path_str.split('/')
        if self.path_str.startswith('/'):
            items[0] = '/'
        return EPath(items[item])


    def imread_gs_int(self, **kwds):
        return image.imread_gs_int(self.path_str, **kwds)

    def imwrite(self, img):
        r = cv2.imwrite(self.path_str, img)
        msg = "Could not write img at {}".format(self.path_str)
        assert self.exists(), msg

    def write(self, content, binary=""):
        mode = "w{}".format(binary)
        with open(self.path_str, mode=mode) as fd:
            fd.write(content)

    def write_csv(self, csv_content):
        if self.has_suffix():
            fname = self.replace_suffix(".csv").string()
        else:
            fname = self.add_suffix(".csv").string()
        with open(fname, mode="w") as fd:
            fd.write(csv_content)

    def writedf_tocsv(self, df, sep=";"):
        if self.has_suffix():
            fname = self.replace_suffix(".csv").string()
        else:
            fname = self.add_suffix(".csv").string()
        df.to_csv(fname, sep=sep)

    def write_tex(self, tex_content, mode="w"):
        if self.has_suffix():
            fname = self.replace_suffix(".tex").string()
        else:
            fname = self.add_suffix(".tex").string()
        with open(fname, mode=mode) as fd:
            fd.write(tex_content)

    def copyto(self, dir):
        """copy the file at self.path_str to the new directory"""
        if EPath(dir).is_dir():
            if self.is_file():
                copyfile(self.path_str, self.replace_parents(dir).string())
            else:
                raise ValueError("{} is not a file".format(self.string()))
        else:
            raise ValueError("cannot copy, not a dir")



    def __str__(self):
        return "{}".format(self.path_str)
    __repr__ = __str__


# class WPath(EPath):
#     def __init__(self, path_str):
#         EPath.__init__(self, path_str)
#
#     def add_param(self, psuffix, sep='_', obj=True):
#         """add parameters suffix after stem"""
#         # if dict then name params in psuffix
#         # else just put values
#         if isinstance(psuffix, dict):
#             keys = list(map(str, psuffix.keys()))
#             values = list(map(str, psuffix.values()))
#             ssuffix = "_".join("".join(item) for item in zip(keys, values))
#         elif isinstance(psuffix, list) or isinstance(psuffix, tuple):
#             ssuffix = "_".join(map(str, psuffix))
#         else:
#             raise ValueError("psuffix has not the right type")
#
#         return self.add_after_stem(ssuffix, obj=obj)


class NodePath:
    def __init__(self, name, ep, parent=None, children=tuple()):
        self.name = name
        self.ep = EPath(ep)
        self.parent = parent
        setattr(self, self.parent.name, self.parent)

        self.children = list(children)
        for child in self.children:
            setattr(self, self.name.string(), child)

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        self.children = children

    def add_parent(self, parent):
        self.parent = parent




class TreePath:
    def __init__(self, root, children=None):
        self.root = root
        self.children = children


def do():
    wp = EPath("/tmp/kodim20.jpeg")
    # kwd = {
    #     "delta" : 20,
    #     "L" : 2,
    #     "n" : 128,
    # }
    # marked_name = wp.add_param(kwd).add_before_stem("test")
    # print(wp)
    # print(marked_name)
    # print(wp.add_before_stem("hello"))
    #
    # attacked_name = wp.add_param({"jpeg":80, "contrast":1.02})
    # print(attacked_name)

    print(wp.add_suffix(".pjp"))
    print(wp.replace_suffix(".opopo"))
    print(wp.has_suffix())
    print(EPath("/tmp/kodim20").has_suffix())

def do1():
    class C:
        def __init__(self):
            self.a = 0
            self.b = "jkl"

        def __getattr__(self, item):
            pass
            # if hasattr(self, item):
            #     print(item, 'exist')
            # else:
            #     print("no no no")

    inst = C()
    # r0 = inst.a
    r = inst.c
    print(r)


class Project:
    def __init__(self, epaths):
        """builds a tree project for saving files (data, images, etc)"""
        self.epath = epaths

    def mkdir_all(self):
        for ep in self.epath:
            ep.mkdir()










def main():
    do1()


if __name__ == '__main__':
    sys.exit(main())