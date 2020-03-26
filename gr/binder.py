#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Binder class
#
# Author:      kol
#
# Created:     06.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------
import os
import weakref
import logging

class NBinder(object):
    """ Supplementary class to manage widget bindings.

        Tkinter management of event binding doesn't properly manage registration/deregistraion
        of multiple event consumers This class allows to handle such situations properly by
        keeping all callbacks registered to particular event and re-binding them if one gets
        revoked.
        In addition, it allows to register and bind to a custom events the same way as Tkinter
        does. For example, a dialog could raise <Close> event when it's been closed allowing
        consumers to get user input provided in the dialog.
    """
    # Bindings registered in all instances of NBinder
    # TODO: Add threads safety
    __bindings = []

    # binder ID
    __count = 0

    def __init__(self):
        self.bnd_ref = dict()
        self.id = NBinder.__count
        NBinder.__count += 1
        logging.debug("New binder instance id {}".format(self.id))

    def bind(self, widget, event, callback, _type = "tk", add = ''):
        """Bind a callback to an event (tkInter or custom)

        Parameters:
            widget      A Tkinter widget or custom object which has winfo_id() method
            event       A string specifying an event.
                        Tkinter events listed in documentaion. Custom events are specific to objects
                        and should be described in corresponding documentation.
            callback    A callback function which is to be called when event is raised.
                        It should accept one parameter (event data). Returns are ignored.
            _type       "tk" for Tkinter event, anything else for custom
            add         Additional parameter for Tkinter bind() call
        """
        # Make a binding
        wkey = str(widget.winfo_id()) + '__' + str(event)
        logging.debug("Binder {} binding {} to event {}".format(self.id, callback, wkey))
        if _type == "tk":
            # tkinter event
            bnd_id = widget.bind(event, callback, add)
        else:
            # custom event
            bnd_id = len(self.bnd_ref) + 1

        # Save binding
        self.bnd_ref[wkey] = [bnd_id,
            weakref.ref(widget),
            event,
            weakref.WeakMethod(callback),
            _type]

        # Add finalizing callback
        widget.__f = weakref.finalize(widget, self.__finalize, wkey)

        # Store binding globally
        NBinder.__bind(self, widget, wkey, event, callback, _type)
        return bnd_id

    def register(self, widget, event, callback):
        """Bind a callback to a custom event. See bind() for parameters"""
        self.bind(widget, event, callback, "custom")

    def unbind(self, widget, event):
        """Unbind from widget's event.
        If there are several callbacks bound to this Tkinter widget, they gets rebound
        """
        key = str(widget.winfo_id()) + '__' + str(event)
        logging.debug("Binder {} unbinding from event key {}".format(self.id, key))
        if key in self.bnd_ref:
            bnd_id, wref, event, cref, _type = self.bnd_ref[key]
            if _type == "tk":
                try:
                    wref().unbind(event, bnd_id)
                except:
                    pass
            del self.bnd_ref[key]

        NBinder.__unbind(self, widget, None, event)

    def unbind_widget(self, widget):
        """Unbind all callbacks bound to any widget events"""
        logging.debug("Binder {} unbinding from widget {}".format(self.id, widget))
        to_remove = []
        for key in self.bnd_ref.keys():
            bnd_id, wref, event, cref, _type = self.bnd_ref[key]
            if wref() is not None and wref() == widget:
                if _type == "tk":
                    try:
                        wref().unbind(event, bnd_id)
                    except:
                        pass
                to_remove.extend([key])
        for key in to_remove: del self.bnd_ref[key]

        NBinder.__unbind(self, widget, None, None)

    def unbind_key(self, wkey):
        """Unbind all callbacks bound to any widget events"""
        if wkey in self.bnd_ref:
            logging.debug("Binder {} unbinding from key {}".format(self.id, wkey))
            bnd_id, wref, event, cref, _type = self.bnd_ref[wkey]
            if wref() is not None:
                if _type == "tk":
                    try:
                        wref().unbind(event, bnd_id)
                    except:
                        pass
                del self.bnd_ref[wkey]
            NBinder.__unbind(self, None, wkey, None)

    def unbind_all(self):
        """Unbind all registrations made by this instance"""
        logging.debug("Binder {} unbinding all events".format(self.id))
        for key in self.bnd_ref.keys():
            bnd_id, wref, event, cref, _type = self.bnd_ref[key]
            if wref() is not None and _type == "tk":
                try:
                    wref().unbind(event, bnd_id)
                except:
                    pass

        self.bnd_ref = dict()
        NBinder.__unbind(self, None, None, None)

    def __finalize(self, wkey):
        logging.debug("Binder {} finalize for key {}".format(self.id, wkey))
        self.unbind_key(wkey)

    def trigger(self, widget, event, evt_data):
        """Triggers a custom event by calling all registered callbacks

        Parameters:
            widget      A Tkinter widget
            event       An event string
            evt_data    Any event data (usually of Event class)
        """
        for bnd in NBinder.__bindings:
            _owner, _wref, _wkey, _event, _cref, _type = bnd
            if _wref() is not None and _cref() is not None:
                if _wref() == widget and _event == event:
                    logging.debug("Binder {} trigger event {}.{} for {}".format(self.id, _wref(), event, _cref))
                    _cref()(evt_data)

    @staticmethod
    def __bind(owner, widget, wkey, event, callback, _type):
        NBinder.__bindings.extend([
            [owner,
            weakref.ref(widget),
            wkey,
            event,
            weakref.WeakMethod(callback),
            _type]])
        logging.debug("Global bindings after bind = {}".format(len(NBinder.__bindings)))

    @staticmethod
    def __unbind(owner, widget, wkey, event):

        def rebind(wref, event):
            # Find last event subscription and renew it
            logging.debug("Global bindings before rebind = {}".format(len(NBinder.__bindings)))
            for bnd in reversed(NBinder.__bindings):
                _owner, _wref, _wkey, _event, _cref, _type = bnd
                if _wref() == wref() and _event == event and _cref() is not None and _type == 'tk':
                    try:
                        _wref().bind(event, _cref())
                        logging.debug("Global rebinding {}.{} to {}".format(wref(), event, _cref()))
                    except:
                        pass

                    break

        for i, bnd in enumerate(NBinder.__bindings):
            _owner, _wref, _wkey, _event, _cref, _type = bnd

            # Options:
            #   1. Unbind all binder instance registrations:
            #       owner is not None, other params are none
            #   2. Unbind all widget registrations by binder instance:
            #       owner is not None, widget or wkey is not None, event is None
            #   3. Unbind widget event registration by binder instance:
            #       all params not None
            proceed = False
            if event is None and widget is not None:
                # Option 2, widget provided
                proceed = (owner == _owner and _wref() is not None and widget == _wref())
            elif event is None and wkey is not None:
                # Option 2, registration key provided
                proceed = (owner == _owner and wkey == _wkey)
            elif event is not None and widget is not None:
                # Option 3, widget provided
                proceed = (owner == _owner and _event == event and _wref() is not None and widget == _wref())
            elif event is not None and wkey is not None:
                # Option 3, registration key provided
                proceed = (owner == _owner and _event == event and wkey == _wkey)
            else:
                # Option 1
                proceed = (owner == _owner)

            if proceed:
                logging.debug("Global unbind {} from event {}".format(
                    _wref() if _wref() is not None else _wkey, _event))
                del NBinder.__bindings[i]
                if _wref() is not None and _type == 'tk': rebind(_wref, _event)

        logging.debug("Global bindings after unbind = {}".format(len(NBinder.__bindings)))


