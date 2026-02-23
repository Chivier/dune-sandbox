import os
import json
import datetime
import secrets
import tornado
import tornado.ioloop
import tornado.web
import tornado.websocket
import util

util.parsed_args.filter = "context"

# room_id -> room_chat_history
Database = {}
# uid -> set(room_id). Use a set to avoid duplicates; convert to list when serializing.
users = {}
rooms = {} # room_id -> {"owner": uid, "name": name, "visibility": visibility}
accounts = {} # uid -> {"pwd": pwd}
tokens = {} # token -> {"uid": uid}
invites = {} # invite_token -> {"room_id": room_id, "uid": uid}
message_counter = 0
# room_id -> set(WebSocketHandler) for live room updates
room_streams = {}
# uid -> set(WebSocketHandler) for live agent tool-call trace updates
agent_streams = {}

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def append_message(room_id, user_id, message):
    global message_counter
    if room_id not in Database:
        Database[room_id] = []
    ensure_user_room(user_id, room_id)
    message_counter += 1
    msg_obj = {
        "message_id": message_counter,
        "created_at": now_iso(),
        "user_id": user_id,
        "message": message,
    }
    Database[room_id].append(msg_obj)
    return msg_obj

def broadcast_room_event(room_id, event):
    payload = json.dumps(event)
    conns = room_streams.get(room_id, set())
    dead = []
    for conn in list(conns):
        try:
            conn.write_message(payload)
        except Exception:
            dead.append(conn)
    for conn in dead:
        conns.discard(conn)

def broadcast_agent_event(uid, event):
    payload = json.dumps(event)
    conns = agent_streams.get(uid, set())
    dead = []
    for conn in list(conns):
        try:
            conn.write_message(payload)
        except Exception:
            dead.append(conn)
    for conn in dead:
        conns.discard(conn)

def get_history(room_ids):
    history = {}
    for room_id in room_ids:
        if room_id in Database:
            history[room_id] = Database[room_id]
        else:
            history[room_id] = []
    return history

def get_history_for_user(user_id):
    # compose history for all rooms the user is in
    if user_id in users:
        room_ids = users[user_id]
        return get_history(room_ids)

def ensure_user_room(uid, room_id):
    if uid not in users:
        users[uid] = set()
    users[uid].add(room_id)

def make_uid():
    # Human-readable, URL-safe, and unlikely to collide.
    # Keep uid simple because it's used in room_id formatting.
    while True:
        uid = f"user_{secrets.token_hex(4)}"
        if uid not in accounts:
            return uid

def make_token(uid):
    # Bearer-ish token: random and unguessable.
    # Note: we don't embed the uid to avoid leaking it.
    while True:
        token = f"tok_{secrets.token_urlsafe(24)}"
        if token not in tokens:
            return token

def make_invite(uid, room_id):
    # Invite tokens should also be unguessable.
    while True:
        invite_token = f"inv_{secrets.token_urlsafe(24)}"
        if invite_token not in invites:
            return invite_token

class Basic_handler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Tornado!")

class Base_handler(tornado.web.RequestHandler):
    def prepare(self):
        token = self.get_argument("token", None)
        if token is None or token not in tokens:
            self.set_status(401)
            self.write({"status": "fail"})
            self.finish()
            return

        uid = self.get_argument("uid", None)
        if uid is None:
            uid = self.get_argument("user_id", None)
        if uid is not None and tokens[token]["uid"] != uid:
            self.set_status(401)
            self.write({"status": "fail"})
            self.finish()
            return

class Chat_handler(tornado.web.RequestHandler):
    def get(self):
        room_id = self.get_argument("room_id")
        if room_id in Database:
            self.write({"room_id": room_id, "messages": Database[room_id]})
        else:
            self.write({"room_id": room_id, "messages": []})

    def post(self):
        # maybe websocket for real-time chat later
        room_id = self.get_argument("room_id")
        user_id = self.get_argument("user_id")
        message = self.get_argument("message")

        msg_obj = append_message(room_id, user_id, message)
        broadcast_room_event(room_id, {"type": "room_message", "room_id": room_id, "message": msg_obj})
        self.write({"status": "success", "room_id": room_id, "messages": Database[room_id]})

class Agent_handler(Base_handler):
    async def post(self):
        # placeholder for agent-related logic
        user_id = self.get_argument("user_id")
        msg = self.get_argument("message")
        rulebook_id = (self.get_argument("rulebook_id", "hipaa") or "hipaa").strip().lower()
        rulebook_text = self.get_argument("rulebook_text", "")
        run_id = (self.get_argument("run_id", "") or "").strip()
        if not run_id:
            run_id = f"run_{secrets.token_hex(8)}"
        room_ids_raw = self.get_argument("room_ids", "")
        room_ids = [room_id.strip() for room_id in room_ids_raw.split(",") if room_id.strip()]
        if not room_ids:
            self.set_status(400)
            self.write({"status": "fail"})
            return
        # get the agent action and response from some agent system
        # here it's just a mockup
        history = get_history(room_ids)
        print(history)
        loop = tornado.ioloop.IOLoop.current()

        def emit_agent_event(event):
            payload = dict(event or {})
            payload.setdefault("uid", user_id)
            payload.setdefault("run_id", run_id)
            loop.add_callback(broadcast_agent_event, user_id, payload)

        emit_agent_event({"type": "run_start"})
        try:
            response = await loop.run_in_executor(
                None,
                lambda: util.one_liner(
                    f"the message is {user_id} : {msg}, avalible history: {history}",
                    rulebook_id=rulebook_id,
                    rulebook_text=rulebook_text,
                    on_tool_call=lambda tool_event: emit_agent_event(
                        {"type": "tool_call", **(tool_event or {})}
                    ),
                ),
            )
        except ValueError as exc:
            self.set_status(400)
            self.write({"status": "fail", "error": str(exc)})
            emit_agent_event({"type": "run_end", "status": "fail"})
            return
        except Exception as exc:
            self.set_status(500)
            self.write({"status": "fail", "error": str(exc)})
            emit_agent_event({"type": "run_end", "status": "fail"})
            return
        emit_agent_event({"type": "run_end", "status": "success"})
        self.write({"status": "success", "response": response})

class Room_invite_handler(Base_handler):
    def post(self):
        uid = self.get_argument("uid")
        room_id = self.get_argument("room_id")
        invite_token = make_invite(uid, room_id)
        invites[invite_token] = {"room_id": room_id, "uid": uid}
        self.write({"invite_token": invite_token, "expires_at": None})

class Room_join_handler(Base_handler):
    def post(self):
        uid = self.get_argument("uid")
        invite_token = self.get_argument("invite_token")
        invite = invites.get(invite_token)
        if invite is None:
            self.set_status(404)
            self.write({"status": "fail"})
            return
        room_id = invite.get("room_id")
        if room_id is None:
            self.set_status(500)
            self.write({"status": "fail"})
            return
        ensure_user_room(uid, room_id)
        self.write({"status": "success", "room_id": room_id})

class Room_talk_handler(Base_handler):
    def post(self):
        uid = self.get_argument("uid")
        room_id = self.get_argument("room_id")
        msg = self.get_argument("msg")

        msg_obj = append_message(room_id, uid, msg)
        broadcast_room_event(room_id, {"type": "room_message", "room_id": room_id, "message": msg_obj})
        self.write({"message_id": msg_obj["message_id"], "created_at": msg_obj["created_at"]})

class Room_create_handler(Base_handler):
    def post(self):
        uid = self.get_argument("uid")
        name = self.get_argument("name")
        visibility = self.get_argument("visibility", None)
        room_id = f"room_{uid}_{len(rooms)}"
        rooms[room_id] = {"owner": uid, "name": name, "visibility": visibility}
        if room_id not in Database:
            Database[room_id] = []
        ensure_user_room(uid, room_id)
        self.write({"room_id": room_id})

class Room_history_handler(Base_handler):
    def get(self):
        room_id = self.get_argument("room_id")
        self.write({"messages": Database.get(room_id, [])})

class Room_info_handler(Base_handler):
    def get(self):
        room_id = self.get_argument("room_id")
        info = rooms.get(room_id)
        if info is None:
            self.set_status(404)
            self.write({"status": "fail"})
            return
        self.write({
            "room_id": room_id,
            "owner": info.get("owner"),
            "name": info.get("name"),
            "visibility": info.get("visibility"),
        })

class Room_stream_handler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        # This console is typically served from the same Tornado instance, but allow
        # browser-based development across localhost ports.
        return True

    def open(self):
        token = self.get_argument("token", None)
        if token is None or token not in tokens:
            self.close(code=4401, reason="unauthorized")
            return

        uid = self.get_argument("uid", None)
        if uid is not None and tokens[token]["uid"] != uid:
            self.close(code=4401, reason="unauthorized")
            return

        room_id = self.get_argument("room_id", None)
        if room_id is None or room_id == "":
            self.close(code=4400, reason="room_id required")
            return

        self.room_id = room_id
        if room_id not in room_streams:
            room_streams[room_id] = set()
        room_streams[room_id].add(self)
        self.write_message(json.dumps({"type": "hello", "room_id": room_id}))

    def on_close(self):
        room_id = getattr(self, "room_id", None)
        if not room_id:
            return
        conns = room_streams.get(room_id)
        if conns:
            conns.discard(self)

class Agent_stream_handler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        token = self.get_argument("token", None)
        if token is None or token not in tokens:
            self.close(code=4401, reason="unauthorized")
            return

        uid = self.get_argument("uid", None)
        if uid is None or uid == "":
            uid = tokens[token]["uid"]
        elif tokens[token]["uid"] != uid:
            self.close(code=4401, reason="unauthorized")
            return

        self.uid = uid
        if uid not in agent_streams:
            agent_streams[uid] = set()
        agent_streams[uid].add(self)
        self.write_message(json.dumps({"type": "hello", "uid": uid}))

    def on_close(self):
        uid = getattr(self, "uid", None)
        if not uid:
            return
        conns = agent_streams.get(uid)
        if conns:
            conns.discard(self)

class Account_login_handler(tornado.web.RequestHandler):
    def post(self):
        uid = self.get_argument("uid")
        pwd = self.get_argument("pwd")
        if uid in accounts and accounts[uid]["pwd"] == pwd:
            token = make_token(uid)
            tokens[token] = {"uid": uid}
            self.write({"token": token, "expires_at": None})
        else:
            self.write({"status": "fail"})

class Account_register_handler(tornado.web.RequestHandler):
    def post(self):
        uid = self.get_argument("uid", None)
        pwd = self.get_argument("pwd")
        if uid is None or uid == "":
            uid = make_uid()
        accounts[uid] = {"pwd": pwd}
        self.write({"uid": uid})

class Account_get_rooms_handler(Base_handler):
    def get(self):
        uid = self.get_argument("uid")
        room_ids = list(users.get(uid, set()))
        room_ids.sort()
        self.write({"room_ids": room_ids})

def make_app():
    static_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "front_end"))
    return tornado.web.Application([
        (r"/", Basic_handler),
        (r"/ui", tornado.web.RedirectHandler, {"url": "/ui/"}),
        (r"/ui/(.*)", tornado.web.StaticFileHandler, {"path": static_root, "default_filename": "ui.html"}),
        (r"/chat", Chat_handler),
        (r"/agent", Agent_handler),
        (r"/agent/stream", Agent_stream_handler),
        (r"/room/invite", Room_invite_handler),
        (r"/room/join", Room_join_handler),
        (r"/room/stream", Room_stream_handler),
        (r"/room/talk", Room_talk_handler),
        (r"/room/create", Room_create_handler),
        (r"/room/history", Room_history_handler),
        (r"/room/info", Room_info_handler),
        (r"/account/login", Account_login_handler),
        (r"/account/register", Account_register_handler),
        (r"/account/get_rooms", Account_get_rooms_handler),
    ])

if __name__ == "__main__":

    PORT = 8888

    app = make_app()
    app.listen(PORT)
    print(f"Tornado server started on http://localhost:{PORT}")
    tornado.ioloop.IOLoop.current().start()
