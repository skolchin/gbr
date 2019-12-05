#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Log display modal window
#
# Author:      kol
#
# Created:     03.09.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

import os
import sys
import logging

import tkinter as tk
from io import StringIO
from .ui_extra import GrDialog

class GrLogDlg(GrDialog):
    def __init__(self, root, *args, **kwargs):
        self._log = None
        GrDialog.__init__(self, root, *args, **kwargs)

    def get_minsize(self):
        return (300, 300)

    def get_title(self):
        return "Log"

    def init_params(self, args, kwargs):
        log_list = kwargs.pop("log_list", None)
        log_string =  kwargs.pop("log_string", None)

        if not log_string is None:
            self._log = log_string.splitlines()
        elif not log_list is None:
            self._log = log_list
        else:
            raise Exception("Log not provided")

    def init_frame(self):
        sbr = tk.Scrollbar(self.internalFrame)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        sbb = tk.Scrollbar(self.internalFrame, orient=tk.HORIZONTAL)
        sbb.pack(side=tk.BOTTOM, fill=tk.X)

        max_len = min(len(max(self._log, key = lambda f: len(f))), 60)

        lbox = tk.Listbox(self.internalFrame, yscrollcommand=sbr.set, xscrollcommand=sbb.set,
            width=max_len)
        lbox.insert(tk.END, *self._log)
        lbox.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)

        for i, item in enumerate(self._log):
            if item.startswith("WARNING"):
                lbox.itemconfig(i, {'fg': 'green'})
            elif item.startswith("ERROR"):
                lbox.itemconfig(i, {'fg': 'red'})

        sbr.config(command = lbox.yview)
        sbb.config(command = lbox.xview)

    def grab_focus(self):
        self.focus_set()
        self.grab_set()

    def init_buttons(self):
        tk.Button(self.buttonFrame, text = "Close",
            command = self.close_click_callback).pack(side = tk.TOP, padx = 5, pady = 5)
        self.buttonFrame.configure(bd = 0, relief = tk.FLAT)

class GrLog(object):
    """GBR logging class"""

    class GrLogFilter(logging.Filter):
        """ Enclosed log filter class"""
        def __init__(self, name=''):
            logging.Filter.__init__(self, name)
            self.n_errors = 0

        def filter(self, record):
            f = logging.Filter.filter(self, record)
            if f and record.levelno == logging.ERROR:
                self.n_errors += 1
            return f

    def __init__(self, level = logging.INFO):
        """Initialize logging system"""
        self.log_stream = StringIO()
        self.log_filter = self.GrLogFilter()
        self.dlg = None
        logging.basicConfig (stream=self.log_stream, format='%(levelname)s: %(message)s', level=level)
        logging.getLogger().addFilter(self.log_filter)

    def _get(self):
        """Returns current log entries as array of strings"""
        return self.log_stream.getvalue().splitlines()

    def _show(self, root):
        """Show Log info dialog. If no log written, returns False"""
        log = self.log_stream.getvalue()
        if not log is None and len(log) > 0:
           dlg = GrLogDlg(root, log_string = log)
           return True
        else:
           return False

    def _clear(self):
        """Clear log"""
        self.log_stream.truncate(0)
        self.log_stream.seek(0)
        self.log_filter.n_errors = 0

    def _getNumErrors(self):
        """Returns number of errros passed to the log"""
        return self.log_filter.n_errors

    # Static methods
    __log = None

    @staticmethod
    def init():
        if GrLog.__log is None: GrLog.__log = GrLog()
        return GrLog.__log

    @staticmethod
    def get():
        return GrLog.__log._get()

    @staticmethod
    def show(root):
        return GrLog.__log._show(root)

    @staticmethod
    def clear():
        GrLog.__log._clear()

    @staticmethod
    def numErrors():
        return GrLog.__log._getNumErrors()

