"""
session
~~~~~~~

Stores recent sessions
"""

import open_cp.gui.locator as locator
import open_cp.gui.tk.session_view as session_view

class Session():
    def __init__(self, root=None):
        self._root = root
        self.model = SessionModel()
        self.filename = None
    
    def run(self):
        self.view = session_view.SessionView(self._root, self, self.model)
        self.view.wait_window(self.view)
        return self.filename
    
    def selected(self, index):
        self.filename = self.model.recent_sessions[index]
        self.view.cancel()
        
    def new_session(self, filename):
        self.model.new_session(filename)


class SessionModel():
    def __init__(self):
        self.settings  = locator.get("settings")
        self._populate_sessions()

    def _populate_sessions(self):
        sessions = []
        for key in self._session_keys():
            num = int(key[7:])
            sessions.append((num, self.settings[key]))
        sessions.sort(key = lambda p : p[0])                
        self._sessions = [p[1] for p in sessions]
        
    def _session_keys(self):
        return [key for key in self.settings if key.startswith("session")]
    
    @property
    def recent_sessions(self):
        return list(self._sessions)
    
    def save(self):
        for key in self._session_keys():
            del self.settings[key]
        for index, name in enumerate(self._sessions):
            self.settings["session{}".format(index)] = name
        self.settings.save()
    
    def new_session(self, filename):
        try:
            index = self._sessions.index(filename)
            del self._sessions[index]
            self._sessions.insert(0, filename)
        except ValueError:
            self._sessions.insert(0, filename)
            self._sessions = self._sessions[:10]
                
        self.save()
        