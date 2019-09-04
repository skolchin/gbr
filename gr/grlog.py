#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Log display modal window
#
# Author:      skolchin
#
# Created:     03.09.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import os
import sys
import logging

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
    import tkSimpleDialog as simpledialog
    from cStringIO import StringIO
else:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import simpledialog
    from io import StringIO

class GrLogWindow(simpledialog.Dialog):
      def __init__(self, parent, log_list = None, log_string = None):
          self._log = None
          if not log_string is None:
             self._log = log_string.splitlines()
          elif not log_list is None:
               self._log = log_list
          else:
             raise Exception("Log not provided")

          simpledialog.Dialog.__init__(self, parent, "Log")

      def body(self, master):
        self.bodyMaster = master

        sbr = tk.Scrollbar(master)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        sbb = tk.Scrollbar(master, orient=tk.HORIZONTAL)
        sbb.pack(side=tk.BOTTOM, fill=tk.X)

        max_len = min(len(max(self._log, key = lambda f: len(f))),50)

        self.lbox = tk.Listbox(master, yscrollcommand=sbr.set, xscrollcommand=sbb.set, width=max_len)
        self.lbox.insert(tk.END, *self._log)
        self.lbox.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)

        sbr.config(command=self.lbox.yview)
        sbb.config(command=self.lbox.xview)

        return self.lbox

      def buttonbox(self):
        box = tk.Frame(self)

        w = tk.Button(box, text="Close", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.TOP, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.ok)

        box.pack(side = tk.TOP, fill = tk.Y, expand = True)
        self.bodyMaster.pack_configure(fill = tk.BOTH, expand = True)


class GrLog(object):
    """GBR logging system"""

    def __init__(self, level = logging.INFO):
        """Initialize logging system"""
        self.log_stream = StringIO()
        logging.basicConfig (stream=self.log_stream, format='%(levelname)s: %(message)s', level=level)

    def getLog(self):
        """Returns current log entries as array of strings"""
        return self.log_stream.getvalue().splitlines()

    def showLog(self, root):
        """Show Log info dialog. If no log written, returns False"""
        log = self.log_stream.getvalue()
        if not log is None and len(log) > 0:
           dlg = GrLogWindow(root, log_string = log)
           return True
        else:
           return False

    def clearLog(self):
        """Clear log"""
        self.log_stream.truncate(0)
        self.log_stream.seek(0)

    __log = None

    @staticmethod
    def init():
        if GrLog.__log is None: GrLog.__log = GrLog()
        return GrLog.__log

    @staticmethod
    def get():
        return GrLog.__log.getLog()

    @staticmethod
    def show(root):
        return GrLog.__log.showLog(root)

    @staticmethod
    def clear():
        GrLog.__log.clearLog()

