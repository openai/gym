# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2020 pyglet contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
"""
Responsabilities

    Defines the events that modify media_player state
    Defines which events are potential defects
    Gives the low level support to extract info from the recorded data
    For new code here, keep accepting and returning only data structures,
    never paths or files.
"""

# events definition
mp_events = {
    "version": 1.1,

    # <evname>: {
    #     "desc": <description used in reports to mention the event>,
    #     "update_names": <list of names of fields updated>,
    #     "other_fields": <list of additionals fields to show when mention the event in a report>
    #     },

    "crash": {
        "desc": "media_player crashed.",
        "update_names": ["evname", "sample"],
        "other_fields": [],
        "test_cases": [("crash", "small.mp4")]
        },

    "mp.im": {
        "desc": "Play",
        "update_names": ["evname", "sample"],
        "other_fields": [],
        "test_cases": [("mp.im", 3, "small.mp4")]
        },

    "p.P._sp": {
        "desc": "Start playing",
        "update_names": ["evname", "wall_time"],
        "other_fields": [],
        "test_cases": [("p.P._sp", 1.23)]
        },

    "p.P.sk": {
        "desc": "Seek",
        "update_names": ["evname", "seek_to_time"],
        "other_fields": [],
        "test_cases": [("p.P.sk", 1.23), ("p.P.sk", None)]
        },

    "p.P.ut.1.0": {
        "desc": "Enter update_texture",
        "update_names": ["evname", "pyglet_dt", "current_time", 
                         "audio_time", "wall_time"],
        "other_fields": [],
        "test_cases": [("p.P.ut.1.0", 0.02, 2.31, 2.28, 1.21),
                       ("p.P.ut.1.0", 0.02, None, 2.28, 1.21),
                       ("p.P.ut.1.0", None, 2.31, 2.28, 1.21)]
        },
    "p.P.ut.1.5": {
        "desc": "Discard video frame too old,",
        "update_names": ["evname", "video_time"],
        "other_fields": ["current_time"],
        "test_cases": [("p.P.ut.1.5", 1.21)]
        },
    "p.P.ut.1.6": {
        "desc": "Current video frame,",
        "update_names": ["evname", "video_time"],
        "other_fields": [],
        "test_cases": [("p.P.ut.1.6", 1.21)]
        },
    "p.P.ut.1.7": {
        "desc": "Early return doing nothing because video_time is None (likely EOV),",
        "update_names": ["evname", "rescheduling_time"],
        "other_fields": [],
        "test_cases": [("p.P.ut.1.7", 0.02)]
        },
    "p.P.ut.1.8": {
        "desc": "Image frame is None (?)",
        "update_names": ["evname"],
        "other_fields": [],
        "test_cases": [("p.P.ut.1.8",)]
        },
    # in log_render_anomalies list only if rescheduling_time < 0
    "p.P.ut.1.9": {
        "desc": "Re-scheduling,",
        "update_names": ["evname", "rescheduling_time", "next_video_time"],
        "other_fields": [],
        "test_cases": [("p.P.ut.1.9", 0.02, None), ("p.P.ut.1.9", 0.02, 2.7)]
        },

    # crash_detection relies in this being the last event in the log_entries
    "p.P.oe": {
        "desc": ">>> play ends",
        "update_names": ["evname"],
        "other_fields": [],
        "test_cases": [("p.P.oe",)]
        },
    }

# events to examine for defects detection
mp_bads = {"crash", "p.P.ut.1.5", "p.P.ut.1.7", "p.P.ut.1.8"}


class MediaPlayerStateIterator:
    """Exposes for analysis the sequence of media_player states

    Typical use
        mp_states = MediaPlayerStateIterator()
        for st in mp_states:
            do something with st, the current media_player state.

    If desired a callback can be called just before processing an event, the
    signature is
        fn_pre_event(event, state_before_event)

    The mp state is handled as a dict, with keys in cls.fields
    """
    fields = {
        # real
        "evname": None,
        "evnum": -1,  # synthetic, ordinal last event processed
        "sample": None,
        "wall_time": None,
        "current_time": None,
        "audio_time": None,
        "seek_to_time": None,
        "pyglet_dt": None,
        "video_time": None,
        "rescheduling_time": None,
        "next_video_time": None,
        # synthetics, probably invalid after using seek
        "pyglet_time": 0,
        "frame_num": 0,
        }

    def __init__(self, recorded_events, events_definition=mp_events, fn_pre_event=None):
        self.fn_pre_event = fn_pre_event
        self.state = dict(self.fields)
        self.events_definition = events_definition
        self.iter_events = iter(recorded_events)
        version_args = next(self.iter_events)
        assert version_args == ("version", self.events_definition["version"])

    def __iter__(self):
        return self

    def __next__(self):
        event = next(self.iter_events)
        if self.fn_pre_event is not None:
            self.fn_pre_event(event, self.state)
        event_dict = self.event_as_dict(event)
        self.update(event_dict)
        return self.state

    def event_as_dict(self, event):
        names = self.events_definition[event[0]]["update_names"]
        updated = {a: b for a, b in zip(names, event)}
        return updated

    def update(self, event_dict):
        self.state.update(event_dict)
        self.state["evnum"] += 1
        evname = event_dict["evname"]
        if evname == "p.P.ut.1.0":
            self.state["pyglet_time"] += event_dict["pyglet_dt"]
        elif evname == "p.P.ut.1.5" or evname == "p.P.ut.1.9":
            self.state["frame_num"] += 1


class TimelineBuilder:
    """At each call to player.Player.update_texture we capture selected player
    state, before accepting the changes in the event. This is the same as
    capturing the state at the end of previous update call.
    Output is a sequence of tuples capturing the desired fields.
    Meant to extract info on behalf of other sw, especially visualization.
    """
    def __init__(self, recorded_events, events_definition=mp_events):
        mp = MediaPlayerStateIterator(recorded_events, events_definition, self.pre)
        self.mp_state_iterator = mp
        self.timeline = []

    def pre(self, event, st):
        if event[0] == "p.P.ut.1.0":
            p = (st["wall_time"], st["pyglet_time"], st["audio_time"],
                 st["current_time"], st["frame_num"], st["rescheduling_time"])
            self.timeline.append(p)

    def get_timeline(self):
        """remember video_time and audio_time can be None"""
        # real work is done in rhe callback pre
        for st in self.mp_state_iterator:
            pass
        # The first entry is bogus, because there was no previous call so discard
        return self.timeline[1:]


def timeline_postprocessing(timeline):
    """ Eliminates Nones in timeline so other software don't error.
        Extra lists are built for the vars with nones, each list with one point
        for each None in the form (wall_time, prev_value).
    """
    current_time_nones = []
    audio_time_nones = []
    old_current_time = 0
    old_audio_time = 0
    filtered_timeline = []
    for wall_time, pt, audio_time, current_time, fnum, rt in timeline:
        if current_time is None:
            current_time = old_current_time
            current_time_nones.append((wall_time, old_current_time))
        else:
            current_time_time = current_time

        if audio_time is None:
            audio_time = old_audio_time
            audio_time_nones.append((wall_time, old_audio_time))
        else:
            old_audio_time = audio_time

        filtered_timeline.append((wall_time, pt, audio_time, current_time, fnum, rt))

    return filtered_timeline, current_time_nones, audio_time_nones


# works for buffered log, needs other implementation if unbuffered
def crash_detected(recorded_events):
    crashed = recorded_events[-1][0] != "p.P.oe"
    return crashed


class CountBads:
    """Helper to report anomalies in the media_player states seen when playing
     a sample.

        - provides .anomalies_description, a dict <anomaly>: <description>
        - calling .count_bads(recorded_events) will return a dict of
          anomaly: <count times anomaly detected>
        - preprocessing: ad-hoc prefiltering the events stream for noise reduction
     """
    def __init__(self, events_definition=mp_events, bads=mp_bads):
        self.events_definition = events_definition
        self.bads = bads
        self.anomalies_description = self.build_anomalies_description()

    def build_anomalies_description(self):
        """builds descriptions for the anomalies"""
        d = self.events_definition
        anomalies_description = {evname: d[evname]["desc"] for evname in self.bads}
        anomalies_description["scheduling_in_past"] = "Scheduling in the past"
        return anomalies_description
    
    def preprocessing(self, recorded_events):
        """
        I see all recordings ending with some potential anomalies in the few
        frames just before the '>>> play ends'; visually the play is perfect so
        I assume they are false positives if just at EOF. Deleting the offending
        events (only if near EOL) to reduce noise in summarize.py
        """
        recorded_events = list(recorded_events)
        if (len(recorded_events) > 9 and
                recorded_events[-2][0] == "p.P.ut.1.7" and
                recorded_events[-6][0] == "p.P.ut.1.7" and
                recorded_events[-10][0] == "p.P.ut.1.7"):
            del recorded_events[-10]
            del recorded_events[-6]
            del recorded_events[-2]

        elif (len(recorded_events) > 6 and
              recorded_events[-2][0] == "p.P.ut.1.7" and
              recorded_events[-6][0] == "p.P.ut.1.7"):
            del recorded_events[-6]
            del recorded_events[-2]

        elif len(recorded_events) > 2 and recorded_events[-2][0] == "p.P.ut.1.7":
            del recorded_events[-2]

        return recorded_events

    def count_bads(self, recorded_events):
        """returns counts of anomalies as a dict of anomaly: count

        recorded_events: media_player events recorded while playing a sample

        Notice that 'counters' has one more key than 'bads': "scheduling_in_past"
        """
        recorded_events = self.preprocessing(recorded_events)
        counters = {k: 0 for k in self.bads}
        cnt_scheduling_in_past = 0
        mp_states = MediaPlayerStateIterator(recorded_events, self.events_definition)
        for st in mp_states:
            evname = st["evname"]
            if evname in counters:
                counters[evname] += 1
            elif ("p.P.ut.1.9" and
                  st["rescheduling_time"] is not None and
                  st["rescheduling_time"] < 0):
                cnt_scheduling_in_past += 1
        counters["scheduling_in_past"] = cnt_scheduling_in_past
        return counters
