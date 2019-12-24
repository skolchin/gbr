#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Log display modal window and GrLog supporting class
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
from time import clock
from functools import reduce

class GrLogDlg(GrDialog):
    """Log display dialog"""

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

    def init_frame(self, internalFrame):
        sbr = tk.Scrollbar(internalFrame)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        sbb = tk.Scrollbar(internalFrame, orient=tk.HORIZONTAL)
        sbb.pack(side=tk.BOTTOM, fill=tk.X)

        max_len = min(len(max(self.__log, key = lambda f: len(f))), 60)

        lbox = tk.Listbox(internalFrame, yscrollcommand=sbr.set, xscrollcommand=sbb.set,
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

    def init_buttons(self, buttonFrame):
        tk.Button(buttonFrame, text = "Close",
            command = self.close_click_callback).pack(side = tk.TOP, padx = 5, pady = 5)
        buttonFrame.configure(bd = 0, relief = tk.FLAT)

class GrLogger(object):
    """GBR logging class.

    This class uses Python's logging facility to save all logging output to an internal
    buffer accessible as a single string. Also, it counts all errors which were triggered
    during last run and allows to display log to end users in a dialog.
    """

    class GrLogFilter(logging.Filter):
        """ Enclosed log filter class"""
        def __init__(self, name = ''):
            logging.Filter.__init__(self, name)
            self.errors = 0
            self.last_error = None

        def filter(self, record):
            f = logging.Filter.filter(self, record)
            if f and record.levelno == logging.ERROR:
                self.errors += 1
                self.last_error = record.getMessage()
            return f

    def __init__(self, master = None, name = None, level = logging.INFO, echo = False, ts = False):
        """Initialize logger

        Parameters:
            master      Tk window or widget to be a parent of log dialog
            name        Logger name
            level       Logging level
                        Note that if multiple GrLogger instances
                        are initialized with the same name, loging level will be
                        set by last instance
            echo        If True, all output will also be printed to stderr
            ts          if True, a timestamp will be added to each log line
        """
        self.master = master
        self.__logger = logging.getLogger(name)
        self.__log_stream = StringIO()
        self.__log_filter = self.GrLogFilter()
        self.__log_dlg = None
        self.t = 0

        self.__logger.setLevel(level)
        self.__logger.addFilter(self.__log_filter)
        self.__logger.propagate = False

        handler = logging.StreamHandler(self.__log_stream)
        formatter = logging.Formatter(
            fmt = '%(levelname)s: %(message)s' if not ts
            else '%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

        if echo:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            self.__logger.addHandler(console)

    @property
    def log(self):
        """Log entries as array of strings"""
        return self.__log_stream.getvalue().splitlines()

    @property
    def errors(self):
        """Number of errors counted in last run"""
        return self.__log_filter.errors

    @property
    def last_error(self):
        """Last error message"""
        return self.__log_filter.last_error

    @property
    def logger(self):
        return self.__logger

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
        self.__log_filter.last_error = None

    def __str__(self):
        """Conversion to string for printing"""
        return self.__log_stream.getvalue()

    def debug(self, *args, **kwargs):
        """Sends a debug message to current logger"""
        self.__logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Sends an info message to current logger"""
        self.__logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Sends an error message to current logger"""
        self.__logger.error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """Sends a warning message to current logger"""
        self.__logger.warning(*args, **kwargs)


    def start(self):
        """Saves time of some code execution start"""
        self.t = clock()
        return self.t

    def stop(self):
        """Returns execution time"""
        # Taken from https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
        self.t = clock() - self.t
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda v,b : divmod(v[0],b) + v[1:], [(self.t*1000,),1000,60,60])

