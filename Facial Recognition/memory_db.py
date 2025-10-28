# memory_db.py
import sqlite3, time
from pathlib import Path

DEFAULTS = dict(pref_distance=1.5, pref_angle_deg=30.0, speed_cap=0.6,
                valence_avg=0.0, arousal_avg=0.0, comfort_avg=0.0,
                confidence=0.7)

class MemoryDB:
    def __init__(self, db_path="spot_memory.db"):
        self.path = Path(db_path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons(
          person_id TEXT PRIMARY KEY,
          display_name TEXT,
          owner INT DEFAULT 0,
          consent INT DEFAULT 1,
          first_seen REAL,
          last_seen REAL,
          seen_count INT DEFAULT 0
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles(
          person_id TEXT PRIMARY KEY,
          pref_distance REAL,
          pref_angle_deg REAL,
          speed_cap REAL,
          valence_avg REAL,
          arousal_avg REAL,
          comfort_avg REAL,
          confidence REAL,
          last_update REAL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          person_id TEXT,
          ts REAL,
          distance REAL,
          angle_deg REAL,
          speed REAL,
          emotion_label TEXT,
          valence REAL,
          arousal REAL,
          comfort REAL,
          conf REAL
        )""")
        self.conn.commit()

    # --- person / profile CRUD ---
    def ensure_person(self, person_id, display_name=None, owner=False):
        now = time.time()
        cur = self.conn.cursor()
        cur.execute("SELECT person_id FROM persons WHERE person_id=?", (person_id,))
        if cur.fetchone() is None:
            cur.execute("""INSERT INTO persons(person_id,display_name,owner,first_seen,last_seen,seen_count)
                           VALUES(?,?,?,?,?,?)""",
                        (person_id, display_name, int(owner), now, now, 1))
            # default profile
            cur.execute("""INSERT INTO profiles(person_id,pref_distance,pref_angle_deg,speed_cap,
                           valence_avg,arousal_avg,comfort_avg,confidence,last_update)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (person_id, DEFAULTS['pref_distance'], DEFAULTS['pref_angle_deg'],
                         DEFAULTS['speed_cap'], DEFAULTS['valence_avg'], DEFAULTS['arousal_avg'],
                         DEFAULTS['comfort_avg'], DEFAULTS['confidence'], now))
        else:
            cur.execute("UPDATE persons SET last_seen=?, seen_count=seen_count+1 WHERE person_id=?",
                        (now, person_id))
        self.conn.commit()

    def get_profile(self, person_id):
        cur = self.conn.cursor()
        cur.execute("""SELECT pref_distance,pref_angle_deg,speed_cap,valence_avg,arousal_avg,
                       comfort_avg,confidence FROM profiles WHERE person_id=?""", (person_id,))
        row = cur.fetchone()
        if row:
            keys = ["pref_distance","pref_angle_deg","speed_cap","valence_avg","arousal_avg","comfort_avg","confidence"]
            return dict(zip(keys, row))
        return {**DEFAULTS}

    def upsert_episode(self, person_id, episode):
        cur = self.conn.cursor()
        cur.execute("""INSERT INTO episodes(person_id,ts,distance,angle_deg,speed,
                       emotion_label,valence,arousal,comfort,conf)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (person_id, episode.get('ts', time.time()), episode.get('distance'),
                     episode.get('angle_deg'), episode.get('speed'), episode.get('emotion_label'),
                     episode.get('valence'), episode.get('arousal'), episode.get('comfort'),
                     episode.get('conf')))
        self.conn.commit()

    def update_profile_from_episode(self, person_id, episode,
                                    eta_d=0.2, eta_s=0.2, base_cap=0.6):
        # confidence gating
        if episode.get('conf', 0.0) < 0.5:
            return
        prof = self.get_profile(person_id)
        d = episode.get('distance', prof['pref_distance'])
        comfort = episode.get('comfort', 0.0)  # [-1,1] later from emotion/AUs
        arousal = episode.get('arousal', 0.0)

        # distance: step away when comfort < 0, gently reduce otherwise
        delta_d = +0.2 if comfort < 0 else -0.1
        new_d = (1-eta_d)*prof['pref_distance'] + eta_d*(d + delta_d)
        new_d = float(min(max(new_d, 0.8), 3.0))

        # speed cap: reduce when arousal high; relax toward base when calm
        if arousal > 0.6:
            new_cap = max(0.2, (1-eta_s)*prof['speed_cap'] + eta_s*(base_cap - 0.2))
        else:
            new_cap = min(base_cap, (1-eta_s)*prof['speed_cap'] + eta_s*base_cap)

        # running means
        val = episode.get('valence', prof['valence_avg'])
        aro = episode.get('arousal', prof['arousal_avg'])
        new_val = 0.9*prof['valence_avg'] + 0.1*val
        new_aro = 0.9*prof['arousal_avg'] + 0.1*aro
        new_comf = 0.9*prof['comfort_avg'] + 0.1*comfort

        cur = self.conn.cursor()
        cur.execute("""UPDATE profiles SET pref_distance=?, speed_cap=?, valence_avg=?,
                       arousal_avg=?, comfort_avg=?, last_update=? WHERE person_id=?""",
                    (new_d, new_cap, new_val, new_aro, new_comf, time.time(), person_id))
        self.conn.commit()

    def close(self):
        self.conn.close()
