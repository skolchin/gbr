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
from .ui_extra import GrDialog

import logging
import tkinter as tk
from io import StringIO

class GrLogDlg(GrDialog):
    """Log show dialog"""

    def __init__(self, *args, **kwargs):
        self.__log = None
        GrDialog.__init__(self, *args, **kwargs)

    def get_minsize(self):
        return (300, 300)

    def get_title(self):
        return "Log"

    def init_params(self, args, kwargs):
        log_list = kwargs.pop("log_list", None)
        log_string =  kwargs.pop("log_string", None)

        if not log_string is None:
            self.__log = log_string.splitlines()
        elif not log_list is None:
            self.__log = log_list
        else:
            raise Exception("Log not provided")

    def init_frame(self):
        sbr = tk.Scrollbar(self.internalFrame)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        sbb = tk.Scrollbar(self.internalFrame, orient=tk.HORIZONTAL)
        sbb.pack(side=tk.BOTTOM, fill=tk.X)

        max_len = min(len(max(self.__log, key = lambda f: len(f))), 60)

        lbox = tk.Listbox(self.internalFrame, yscrollcommand=sbr.set, xscrollcommand=sbb.set,
            width=max_len)
        lbox.insert(tk.END, *self.__log)
        lbox.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)

        for i, item in enumerate(self.__log):
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

class GrLogger(object):
    """GBR logging class"""

    class GrLogFilter(logging.Filter):
        """ Enclosed log filter class"""
        def __init__(self, name=''):
            logging.Filter.__init__(self, name)
            self.errors = 0

        def filter(self, record):
            f = logging.Filter.filter(self, record)
            if f and record.levelno == logging.ERROR:
                self.errors += 1
            return f

    def __init__(self, master = None, level = logging.INFO):
        """Initialize logging system"""
        self.master = master
        self.__log_stream = StringIO()
        self.__log_filter = self.GrLogFilter()
        self.__log_dlg = None
        logging.basicConfig (stream=self.__log_stream, format='%(levelname)s: %(message)s', level=level)
        logging.getLogger().addFilter(self.__log_filter)

    @property
    def log(self):
        """Log entries as array of strings"""
        return self.__log_stream.getvalue().splitlines()

    @property
    def errors(self):
        """Number of errors"""
        return self.__log_filter.errors

    def show(self):
        """Show Log info dialog. Returns either a dialog object or None if log is empty"""
        log = self.__log_stream.getvalue()
        if not log is None and len(log) > 0:
            if self.__log_dlg is not None:
                self.__log_dlg.close()
                self.__log_dlg = None
            self.__log_dlg = GrLogDlg(master = self.master, log_string = log)
        return self.__log_dlg

    def clear(self):
        """Clear log"""
        self.__log_stream.truncate(0)
        self.__log_stream.seek(0)
        self.__log_filter.errors = 0


    def __str__(self):
        return self.__log_stream.getvalue()

