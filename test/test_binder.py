from copy import deepcopy
import numpy as np
import gc
import psutil
import os
import logging
import tkinter as tk

import sys
sys.path.append('../')

from gr.binder import NBinder

SZ = 1024 * 1024        # 1 MB

class Test(object):
    count = 0
    buffer = np.zeros(SZ)

    def __init__(self):
        self.buf = deepcopy(Test.buffer)
        self.id = Test.count
        Test.count += 1
        self.binder = NBinder()

    def winfo_id(self):
        return self.id

    def callback(self, event):
        print("=== Callback {} called".format(self.id))

def mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory usage {}".format(process.memory_info().rss / 1024 /  1024))


logging.basicConfig(level = logging.DEBUG)

def test_mem():
    mem_usage()
    input("Press ENTER: ")

    for i in range(0, 100):
        t = Test()
        print("Instance id {}".format(t.winfo_id()))
        t.binder.register(t, 'Event', t.callback)
        t.binder.trigger(t, 'Event', None)
        t.binder.unbind_widget(t)

        #if i % 100 == 0:
        #    gc.collect()

    mem_usage()
    input("Press ENTER: ")


def test_rebind():
    class Test2(Test):
        def __init__(self, parent):
            Test.__init__(self)
            self.parent = parent

        def callback(self, event):
            self.parent.info.configure(text = "Callback {} called".format(self.id+1))
            self.parent.after(3000, self.parent.clear)

    class TestApp(tk.Tk):
        def __init__(self, *args, **kwargs):
            tk.Tk.__init__(self, "Test app")

            self.top_frame = tk.Frame(self, bd = 1, relief = tk.RAISED)
            self.top_frame.pack(side = tk.TOP, fill = tk.X, expand = True, padx = 5, pady = 5)

            tk.Button(self.top_frame, text = "Bind 1", command = self.bind1).pack(side = tk.LEFT, padx = 5, pady = 5)
            tk.Button(self.top_frame, text = "Bind 2", command = self.bind2).pack(side = tk.LEFT, padx = 5)
            tk.Button(self.top_frame, text = "Unbind 1", command = self.unbind1).pack(side = tk.LEFT, padx = 5)
            tk.Button(self.top_frame, text = "Unbind 2", command = self.unbind2).pack(side = tk.LEFT, padx = 5)

            self.bottom_frame = tk.Frame(self, bd = 1, relief = tk.RAISED)
            self.bottom_frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True, padx = 5, pady = 5)

            self.info = tk.Label(self.bottom_frame, text = "Click here")
            self.info.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            self.clicker1 = Test2(self)
            self.clicker2 = Test2(self)

        def bind1(self):
            self.clicker1.binder.bind(self.info, "<Button-1>", self.clicker1.callback)

        def bind2(self):
            self.clicker2.binder.bind(self.info, "<Button-1>", self.clicker2.callback)

        def unbind1(self):
            self.clicker1.binder.unbind(self.info, "<Button-1>")

        def unbind2(self):
            self.clicker2.binder.unbind(self.info, "<Button-1>")

        def clear(self):
            self.info.configure(text = "Click here")

    app = TestApp()
    app.mainloop()


test_mem()
