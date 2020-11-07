# vim:set et sts=4 sw=4:
#
# ibus-deepspeech - Speech recognition engine for IBus
#
# Copyright (c) 2017 Mike Sheldon <elleo@gnu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Optional
from mic_vad_streaming import DictationThread

import gi

gi.require_version("IBus", "1.0")
from gi.repository import IBus

gi.require_version("Gst", "1.0")
from gi.repository import GLib

MODEL_PATH = "/usr/share/mozilla/deepspeech/models/ds-model.pbmm"
SCORER_PATH = "/usr/share/mozilla/deepspeech/models/ds-model.scorer"


class EngineDeepSpeech(IBus.Engine):
    __gtype_name__ = "EngineDeepSpeech"

    def __init__(self) -> None:
        super(EngineDeepSpeech, self).__init__()
        self.recording = False
        self.worker_thread: Optional[DictationThread] = None
        self.__prop_list = IBus.PropList()
        self.__prop_list.append(
            IBus.Property(
                key="toggle-recording",
                icon="audio-input-microphone",
                type=IBus.PropType.TOGGLE,
                state=0,
                label="Toggle speech recognition",
            )
        )

    def do_focus_in(self) -> None:
        self.register_properties(self.__prop_list)

    def do_property_activate(self, prop_name: str, state: IBus.PropState) -> None:
        print(f"activate {prop_name}")
        if prop_name == "toggle-recording":
            self.recording = not self.recording
            print(f"has {self.recording}")
            if self.recording:
                self.start_recognition()
            else:
                self.stop_recognition()

    def start_recognition(self) -> None:
        print("Starting the recognition thread")

        def callback(text: str) -> None:
            print(f"Recognized: {text}")
            if text:
                self.commit_text(IBus.Text.new_from_string(text + ' '))

        self.worker_thread = DictationThread(MODEL_PATH, SCORER_PATH, callback)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop_recognition(self) -> None:
        print("Stopping the recognition thread")
        assert self.worker_thread
        self.worker_thread.request_stop()
