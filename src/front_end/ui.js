			function getAutoBaseUrl() {
				const origin =
					typeof window !== "undefined" && window.location ? window.location.origin : "";
				if (!origin || origin === "null") return "http://localhost:8888";
				return origin;
			}

			function getStoredBaseOverride() {
				const stored = localStorage.getItem("baseUrlOverride");
				return stored && stored.trim() ? stored.trim() : "";
			}

			function getInitialBaseUrl() {
				return getStoredBaseOverride() || getAutoBaseUrl();
			}

				const state = {
					baseUrl: getInitialBaseUrl(),
					uid: "",
					token: "",
					rooms: [],
					selectedRooms: new Set(),
					agentThread: [],
					activeRoom: "",
					activeRoomMessages: [],
					roomMeta: {},
					roomSocket: null,
					roomSocketRoomId: "",
					roomSocketReconnectTimer: null,
					roomSocketRetryMs: 0,
					rulebookId: "hipaa",
					rulebookText: "",
					toolCalls: [],
					activeAgentRunId: "",
					agentStream: null,
					agentStreamReconnectTimer: null,
					agentStreamRetryMs: 0,
				};

			const el = (id) => document.getElementById(id);

			function safeJsonStringify(value, space = 2) {
				try {
					return JSON.stringify(value, null, space);
				} catch (err) {
					return String(value);
				}
			}

			function stripJsonWrapperParens(text) {
				const trimmed = (text || "").trim();
				if (trimmed.startsWith("(") && trimmed.endsWith(")")) {
					const inner = trimmed.slice(1, -1).trim();
					// Only unwrap when it looks like the model's FINISH([...]) payload.
					if (
						(inner.startsWith("[") && inner.endsWith("]")) ||
						(inner.startsWith("{") && inner.endsWith("}"))
					) {
						return inner;
					}
				}
				return trimmed;
			}

			function tryParseJson(text) {
				try {
					return { ok: true, value: JSON.parse(text) };
				} catch (err) {
					return { ok: false, value: null };
				}
			}

			function formatStructuredValue(value) {
				if (value === null || value === undefined) return "";
				if (Array.isArray(value)) {
					if (!value.length) return "";
					const allPrimitive = value.every(
						(item) =>
							item === null ||
							item === undefined ||
							typeof item === "string" ||
							typeof item === "number" ||
							typeof item === "boolean",
					);
					if (allPrimitive) {
						return value.filter((item) => item !== null && item !== undefined).map(String).join("\n");
					}
					return value.map((item) => formatStructuredValue(item)).filter(Boolean).join("\n\n");
				}
				if (typeof value === "object") {
					const keys = Object.keys(value);
					if (!keys.length) return "";
					return keys
						.sort()
						.map((key) => {
							const v = value[key];
							if (v === null || v === undefined) return "";
							if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
								return `${key}: ${v}`;
							}
							return `${key}: ${safeJsonStringify(v)}`;
						})
						.filter(Boolean)
						.join("\n");
				}
				return String(value);
			}

			function normalizeFinalResponseText(text) {
				if (text === null || text === undefined) return "";
				if (typeof text !== "string") return safeJsonStringify(text);
				let cleaned = text.trim();
				cleaned = cleaned.replace(/^final\s*response\s*:\s*/i, "");
				cleaned = cleaned.replace(/^finish\b\s*/i, "");
				return stripJsonWrapperParens(cleaned).trim();
			}

			function extractFinalResponseFromTimeline(timeline) {
				if (!Array.isArray(timeline) || !timeline.length) return "";
				for (let i = timeline.length - 1; i >= 0; i -= 1) {
					const entry = timeline[i];
					if (!entry) continue;
					const role = entry.role ?? entry.type ?? "";
					const roleText = typeof role === "string" ? role.toLowerCase() : String(role).toLowerCase();
					if (!roleText.includes("final")) continue;

					const content = entry.content ?? entry.text ?? entry.message;
					if (content === undefined || content === null) continue;
					return normalizeFinalResponseText(content);
				}
				return "";
			}

			function formatAgentResponse(response) {
				if (response === null || response === undefined) return "";
				if (typeof response === "string") {
					const normalized = normalizeFinalResponseText(response);
					const parsed = tryParseJson(normalized);
					if (parsed.ok) return formatStructuredValue(parsed.value);
					return normalized;
				}
				if (Array.isArray(response)) return formatStructuredValue(response);
				if (typeof response === "object") {
					// Backend sometimes returns a rich object with a `timeline` that contains the final response.
					const timelineText = extractFinalResponseFromTimeline(response.timeline);
					if (timelineText) return formatAgentResponse(timelineText);

					// Common fallbacks (in case backend shape changes).
					const direct =
						response.final_response ??
						response.finalResponse ??
						response.final ??
						response.answer ??
						response.output ??
						response["Final Response"];
					if (direct !== undefined && direct !== null) {
						return formatAgentResponse(direct);
					}

					return formatStructuredValue(response) || safeJsonStringify(response);
				}
				return String(response);
			}

			function uniqueOrdered(values) {
				const seen = new Set();
				const ordered = [];
				(values || []).forEach((value) => {
					if (!seen.has(value)) {
						seen.add(value);
						ordered.push(value);
					}
				});
				return ordered;
			}

			function asRoomDragData(event, roomId) {
				event.dataTransfer.effectAllowed = "copy";
				event.dataTransfer.setData("application/x-rotunda-room", roomId);
				event.dataTransfer.setData("text/plain", roomId);
			}

			function setStatus(message) {
				el("status").textContent = message;
			}

			function setAuthStatus(message) {
				el("auth-status").textContent = message;
			}

			function isAuthenticated() {
				// Most authenticated UI actions require BOTH uid and token.
				return Boolean(state.token && state.uid);
			}

			function syncModalScrollLock() {
				const authOverlay = el("auth-overlay");
				const settingsOverlay = el("settings-overlay");
				const isModalOpen =
					(authOverlay && !authOverlay.hidden) || (settingsOverlay && !settingsOverlay.hidden);
				document.body.style.overflow = isModalOpen ? "hidden" : "";
			}

			function setAuthOverlayVisible(visible) {
				const overlay = el("auth-overlay");
				if (!overlay) return;
				overlay.hidden = !visible;
				syncModalScrollLock();
			}

			function setSettingsOverlayVisible(visible) {
				const overlay = el("settings-overlay");
				if (!overlay) return;
				overlay.hidden = !visible;
				if (visible) {
					el("base-url").value = state.baseUrl;
					el("uid").value = state.uid;
					el("token").value = state.token;
					el("join-invite").value = "";
					el("settings-status").textContent = getStoredBaseOverride()
						? "using saved base override"
						: "using window base url";
				}
				syncModalScrollLock();
			}

			function loadStored() {
				const storedToken = localStorage.getItem("token");
				const storedUid = localStorage.getItem("uid");
				const storedRulebookId = localStorage.getItem("rulebookId");
				const storedRulebookText = localStorage.getItem("rulebookText");
				if (storedToken) state.token = storedToken;
				if (storedUid) state.uid = storedUid;
				if (storedRulebookId) state.rulebookId = storedRulebookId;
				if (storedRulebookText) state.rulebookText = storedRulebookText;
				state.baseUrl = getInitialBaseUrl();
				el("token").value = state.token;
				el("uid").value = state.uid;
				el("auth-uid").value = state.uid;
				el("base-url").value = state.baseUrl;
				const rulebookSelect = el("rulebook-id");
				if (rulebookSelect) rulebookSelect.value = state.rulebookId;
				const rulebookArea = el("rulebook-text");
				if (rulebookArea) rulebookArea.value = state.rulebookText;
				syncRulebookUi();
			}

			function saveStored() {
				localStorage.setItem("token", state.token || "");
				localStorage.setItem("uid", state.uid || "");
				el("token").value = state.token;
				el("uid").value = state.uid;
			}

			function saveRulebookStored() {
				localStorage.setItem("rulebookId", state.rulebookId || "hipaa");
				localStorage.setItem("rulebookText", state.rulebookText || "");
			}

			function syncRulebookUi() {
				const isCustom = String(state.rulebookId || "").toLowerCase() === "custom";
				const area = el("rulebook-text");
				if (area) area.hidden = !isCustom;
			}

			function saveBaseOverride(baseUrl) {
				const value = (baseUrl || "").trim();
				if (value) {
					localStorage.setItem("baseUrlOverride", value);
				} else {
					localStorage.removeItem("baseUrlOverride");
				}
			}

				async function apiRequest(method, path, data) {
					const url = new URL(state.baseUrl + path);
					const options = { method, headers: {} };
					if (method === "GET") {
						Object.entries(data || {}).forEach(([key, value]) => {
							if (value !== undefined && value !== null && value !== "") {
								url.searchParams.set(key, value);
							}
						});
					} else {
						options.headers["Content-Type"] = "application/x-www-form-urlencoded";
						options.body = new URLSearchParams(data || {}).toString();
					}
					const response = await fetch(url.toString(), options);
					const text = await response.text();
					let json = null;
					try {
						json = JSON.parse(text);
					} catch (err) {
						json = null;
					}
					return { status: response.status, text, json };
				}

				function makeWsUrl(path, data) {
					const wsUrl = new URL(path, state.baseUrl);
					wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
					Object.entries(data || {}).forEach(([key, value]) => {
						if (value !== undefined && value !== null && value !== "") {
							wsUrl.searchParams.set(key, value);
						}
					});
					return wsUrl.toString();
				}

				function clearRoomSocketReconnect() {
					if (!state.roomSocketReconnectTimer) return;
					clearTimeout(state.roomSocketReconnectTimer);
					state.roomSocketReconnectTimer = null;
				}

				function closeRoomSocket() {
					clearRoomSocketReconnect();
					const socket = state.roomSocket;
					state.roomSocket = null;
					state.roomSocketRoomId = "";
					if (!socket) return;
					try {
						socket.onopen = null;
						socket.onmessage = null;
						socket.onerror = null;
						socket.onclose = null;
						socket.close();
					} catch (err) {
						// ignore
					}
				}

				function isLiveRoomConnected() {
					return (
						state.roomSocket &&
						state.roomSocket.readyState === WebSocket.OPEN &&
						state.roomSocketRoomId === state.activeRoom
					);
				}

				function scheduleRoomSocketReconnect(roomId) {
					if (!isAuthenticated() || state.activeRoom !== roomId) return;
					if (state.roomSocketReconnectTimer) return;

					const nextDelay = state.roomSocketRetryMs
						? Math.min(10000, Math.round(state.roomSocketRetryMs * 1.7))
						: 500;
					state.roomSocketRetryMs = nextDelay;
					state.roomSocketReconnectTimer = setTimeout(() => {
						state.roomSocketReconnectTimer = null;
						connectRoomSocket();
					}, nextDelay);
					setStatus(`live disconnected (retrying in ${Math.round(nextDelay / 100) / 10}s)`);
				}

					function connectRoomSocket() {
					clearRoomSocketReconnect();

					if (!isAuthenticated() || !state.activeRoom) {
						closeRoomSocket();
						return;
					}

					const roomId = state.activeRoom;
					if (isLiveRoomConnected()) return;

					closeRoomSocket();

					const wsUrl = makeWsUrl("/room/stream", {
						room_id: roomId,
						uid: state.uid,
						token: state.token,
					});
					let socket;
					try {
						socket = new WebSocket(wsUrl);
					} catch (err) {
						scheduleRoomSocketReconnect(roomId);
						return;
					}

					state.roomSocket = socket;
					state.roomSocketRoomId = roomId;

					socket.onopen = () => {
						state.roomSocketRetryMs = 0;
						setStatus("live updates connected");
					};

					socket.onmessage = (event) => {
						let payload = null;
						try {
							payload = JSON.parse(event.data);
						} catch (err) {
							payload = null;
						}
						if (!payload || payload.room_id !== state.activeRoom) return;
						if (payload.type !== "room_message") return;
						const msg = payload.message;
						if (!msg) return;

						const incomingId = msg.message_id;
						if (incomingId) {
							const exists = state.activeRoomMessages.some((m) => m.message_id === incomingId);
							if (exists) return;
						}
						state.activeRoomMessages.push(msg);
						renderHistory(state.activeRoomMessages);
					};

					socket.onerror = () => {
						// onclose will handle retries
					};

						socket.onclose = (event) => {
						if (state.roomSocket === socket) {
							state.roomSocket = null;
							state.roomSocketRoomId = "";
						}
						// If the server explicitly rejects the socket, don't spin on reconnect.
						// (Browser close codes are numeric; we use 4401 for unauthorized.)
						if (event && event.code === 4401) {
							logout();
							return;
						}
						scheduleRoomSocketReconnect(roomId);
						};
					}

					function setToolCallsStatus(message) {
						const node = el("tool-calls-status");
						if (node) node.textContent = message;
					}

					function renderToolCalls() {
						const list = el("tool-calls-list");
						if (!list) return;
						list.innerHTML = "";

						if (!state.toolCalls.length) {
							const empty = document.createElement("div");
							empty.className = "tool-call-item";
							empty.style.opacity = "0.7";
							empty.textContent = "(no tool calls yet)";
							list.appendChild(empty);
							return;
						}

						state.toolCalls.forEach((callText) => {
							const item = document.createElement("div");
							item.className = "tool-call-item";
							item.textContent = String(callText || "");
							list.appendChild(item);
						});
						list.scrollTop = list.scrollHeight;
					}

					function clearAgentStreamReconnect() {
						if (!state.agentStreamReconnectTimer) return;
						clearTimeout(state.agentStreamReconnectTimer);
						state.agentStreamReconnectTimer = null;
					}

					function closeAgentStream() {
						clearAgentStreamReconnect();
						const socket = state.agentStream;
						state.agentStream = null;
						if (!socket) return;
						try {
							socket.onopen = null;
							socket.onmessage = null;
							socket.onerror = null;
							socket.onclose = null;
							socket.close();
						} catch (err) {
							// ignore
						}
					}

					function isAgentStreamConnected() {
						return state.agentStream && state.agentStream.readyState === WebSocket.OPEN;
					}

					function scheduleAgentStreamReconnect() {
						if (!isAuthenticated()) return;
						if (state.agentStreamReconnectTimer) return;

						const nextDelay = state.agentStreamRetryMs
							? Math.min(10000, Math.round(state.agentStreamRetryMs * 1.7))
							: 500;
						state.agentStreamRetryMs = nextDelay;
						state.agentStreamReconnectTimer = setTimeout(() => {
							state.agentStreamReconnectTimer = null;
							connectAgentStream();
						}, nextDelay);
						setToolCallsStatus(
							`disconnected (retrying in ${Math.round(nextDelay / 100) / 10}s)`,
						);
					}

					function connectAgentStream() {
						clearAgentStreamReconnect();
						if (!isAuthenticated()) {
							closeAgentStream();
							setToolCallsStatus("disconnected");
							return;
						}
						if (isAgentStreamConnected()) return;

						closeAgentStream();

						const wsUrl = makeWsUrl("/agent/stream", {
							uid: state.uid,
							token: state.token,
						});

						let socket;
						try {
							socket = new WebSocket(wsUrl);
						} catch (err) {
							scheduleAgentStreamReconnect();
							return;
						}

						state.agentStream = socket;

						socket.onopen = () => {
							state.agentStreamRetryMs = 0;
							setToolCallsStatus("connected");
						};

						socket.onmessage = (event) => {
							let payload = null;
							try {
								payload = JSON.parse(event.data);
							} catch (err) {
								payload = null;
							}
							if (!payload) return;

							if (payload.type === "tool_call") {
								if (payload.uid && payload.uid !== state.uid) return;
								if (state.activeAgentRunId && payload.run_id && payload.run_id !== state.activeAgentRunId) {
									return;
								}
								const step = payload.step ? `#${payload.step} ` : "";
								const callText = payload.call ? String(payload.call) : "";
								if (!callText) return;
								state.toolCalls.push(`${step}${callText}`.trim());
								renderToolCalls();
							} else if (payload.type === "run_start") {
								if (state.activeAgentRunId && payload.run_id && payload.run_id !== state.activeAgentRunId) {
									return;
								}
								setToolCallsStatus("connected • running");
							} else if (payload.type === "run_end") {
								if (state.activeAgentRunId && payload.run_id && payload.run_id !== state.activeAgentRunId) {
									return;
								}
								setToolCallsStatus("connected • idle");
							}
						};

						socket.onerror = () => {
							// onclose handles retries
						};

						socket.onclose = (event) => {
							if (state.agentStream === socket) {
								state.agentStream = null;
							}
							if (event && event.code === 4401) {
								logout();
								return;
							}
							scheduleAgentStreamReconnect();
						};
					}

					function makeRunId() {
						try {
							if (window.crypto && typeof window.crypto.randomUUID === "function") {
								return `run_${window.crypto.randomUUID()}`;
							}
						} catch (err) {
							// fall back below
						}
						return `run_${Date.now()}_${Math.random().toString(16).slice(2)}`;
					}

					function requireAuth() {
						if (!isAuthenticated()) {
							setStatus("login required");
							setSettingsOverlayVisible(false);
						setAuthOverlayVisible(true);
					return false;
				}
				return true;
			}

			function validateRoomId(roomId) {
				if (!roomId) return null;
				if (!state.rooms.includes(roomId)) return null;
				return roomId;
			}

			function getRoomDisplayName(roomId) {
				const meta = state.roomMeta && roomId ? state.roomMeta[roomId] : null;
				const name = meta && typeof meta.name === "string" ? meta.name.trim() : "";
				return name || roomId;
			}

			function formatRoomLabel(roomId) {
				if (!roomId) return "No room selected";
				const displayName = getRoomDisplayName(roomId);
				if (displayName && displayName !== roomId) {
					return `Room: ${displayName} (${roomId})`;
				}
				return `Room: ${roomId}`;
			}

			function updateActiveRoomLabel() {
				el("active-room-label").textContent = formatRoomLabel(state.activeRoom);
			}

			function clearInviteToken() {
				const box = el("invite-box");
				const token = el("invite-token");
				if (token) token.value = "";
				if (box) box.hidden = true;
			}

			function updateInviteControls() {
				const btn = el("invite-room");
				if (!btn) return;
				btn.disabled = !state.activeRoom;
			}

				function setActiveRoom(roomIdRaw) {
					const roomId = validateRoomId(roomIdRaw);
					if (!roomId) return;
					if (state.activeRoom && state.activeRoom !== roomId) {
						closeRoomSocket();
					}
					state.activeRoom = roomId;
					state.activeRoomMessages = [];
					clearInviteToken();
					updateActiveRoomLabel();
					updateInviteControls();
					renderRooms();
					renderHistory([]);
					loadActiveHistory();
				}

			function renderRooms() {
				const list = el("rooms-list");
				list.innerHTML = "";
				state.rooms.forEach((roomId) => {
					const item = document.createElement("div");
					item.className = "rooms-item";
					item.classList.toggle("selected", state.selectedRooms.has(roomId));
					item.classList.toggle("active", state.activeRoom === roomId);
					const title = document.createElement("div");
					title.className = "rooms-title";
					title.textContent = getRoomDisplayName(roomId);
					item.appendChild(title);

					if (getRoomDisplayName(roomId) !== roomId) {
						const subtitle = document.createElement("div");
						subtitle.className = "rooms-subtitle";
						subtitle.textContent = roomId;
						item.appendChild(subtitle);
					}
					item.setAttribute("draggable", "true");
					item.addEventListener("dragstart", (event) => asRoomDragData(event, roomId));
					item.addEventListener("click", () => setActiveRoom(roomId));
					list.appendChild(item);
				});
			}

			function renderSelectedRooms() {
				const wrap = el("selected-rooms");
				wrap.innerHTML = "";
				if (state.selectedRooms.size === 0) {
					const hint = document.createElement("div");
					hint.className = "status";
					hint.textContent = "No rooms in context yet.";
					wrap.appendChild(hint);
					return;
				}
					state.selectedRooms.forEach((roomId) => {
						const pill = document.createElement("span");
						pill.className = "pill";
						const label = document.createElement("span");
						label.textContent = getRoomDisplayName(roomId);
						if (getRoomDisplayName(roomId) !== roomId) {
							label.title = roomId;
						}
						const remove = document.createElement("button");
					remove.type = "button";
					remove.title = "Remove from context";
					remove.setAttribute("aria-label", `Remove ${roomId} from context`);
					remove.textContent = "×";
					remove.addEventListener("click", () => {
						state.selectedRooms.delete(roomId);
						renderSelectedRooms();
						renderRooms();
					});
					pill.appendChild(label);
					pill.appendChild(remove);
					wrap.appendChild(pill);
				});
			}

			function parseRoomMessageTimeMs(msg) {
				if (!msg) return null;
				const iso = msg.created_at ?? msg.createdAt;
				if (!iso) return null;
				const ms = Date.parse(iso);
				return Number.isFinite(ms) ? ms : null;
			}

			function formatRoomMessageTime(iso) {
				if (!iso) return "";
				try {
					const d = new Date(iso);
					if (Number.isNaN(d.getTime())) return "";
					return d.toLocaleString();
				} catch (err) {
					return "";
				}
			}

			function renderHistory(messages) {
				const list = el("history-list");
				if (!list) return;

				const stickToBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 12;
				list.innerHTML = "";

				if (!state.activeRoom) {
					const empty = document.createElement("div");
					empty.className = "history-item";
					empty.textContent = "Pick a room from the left sidebar to view messages.";
					list.appendChild(empty);
					return;
				}

				updateActiveRoomLabel();

				const sorted = (messages || [])
					.map((msg, index) => ({ msg, index }))
					.sort((a, b) => {
						const timeA = parseRoomMessageTimeMs(a.msg);
						const timeB = parseRoomMessageTimeMs(b.msg);
						if (timeA !== null && timeB !== null && timeA !== timeB) return timeA - timeB;
						if (timeA !== null && timeB === null) return -1;
						if (timeA === null && timeB !== null) return 1;

						const idA = a.msg && a.msg.message_id !== undefined ? Number(a.msg.message_id) : null;
						const idB = b.msg && b.msg.message_id !== undefined ? Number(b.msg.message_id) : null;
						const idAOk = idA !== null && Number.isFinite(idA);
						const idBOk = idB !== null && Number.isFinite(idB);
						if (idAOk && idBOk && idA !== idB) return idA - idB;
						if (idAOk && !idBOk) return -1;
						if (!idAOk && idBOk) return 1;
						return a.index - b.index;
					})
					.map((entry) => entry.msg);

				if (!sorted.length) {
					const empty = document.createElement("div");
					empty.className = "history-item";
					empty.style.opacity = "0.7";
					empty.textContent = "(no messages)";
					list.appendChild(empty);
					return;
				}

				let lastUserId = null;
				sorted.forEach((msg) => {
					const userId = msg && msg.user_id ? String(msg.user_id) : "unknown";
					if (userId !== lastUserId) {
						const header = document.createElement("div");
						header.className = "history-item user-header";
						header.textContent = userId;
						list.appendChild(header);
						lastUserId = userId;
					}

					const createdAt = msg ? msg.created_at ?? msg.createdAt : "";
					const timeLabel = formatRoomMessageTime(createdAt);
					const text = msg && msg.message !== undefined && msg.message !== null ? String(msg.message) : "";
					const item = document.createElement("div");
					item.className = "history-item user-message";
					item.textContent = timeLabel ? `[${timeLabel}] ${text}` : text;
					list.appendChild(item);
				});

				if (stickToBottom) list.scrollTop = list.scrollHeight;
			}

			function renderAgentThread() {
				const list = el("agent-thread");
				list.innerHTML = "";
				if (!state.agentThread.length) {
					const empty = document.createElement("div");
					empty.className = "agent-item";
					empty.style.opacity = "0.7";
					empty.textContent = "Ask the agent once you have rooms in context.";
					list.appendChild(empty);
					return;
				}

				state.agentThread.forEach((entry) => {
					const item = document.createElement("div");
					item.className = "agent-item";
					const role = entry && entry.role !== undefined && entry.role !== null ? String(entry.role) : "agent";
					let text = "";
					if (role.toLowerCase() === "agent" || role.toLowerCase() === "assistant") {
						text = formatAgentResponse(entry ? entry.text : "");
					} else if (entry && entry.text !== undefined && entry.text !== null) {
						text = typeof entry.text === "string" ? entry.text : safeJsonStringify(entry.text);
					}
					const separator = text.includes("\n") ? ":\n" : ": ";
					item.textContent = `${role}${separator}${text}`;
					list.appendChild(item);
				});
				list.scrollTop = list.scrollHeight;
			}

			async function refreshRooms() {
				if (!requireAuth()) return;
				const result = await apiRequest("GET", "/account/get_rooms", {
					uid: state.uid,
					token: state.token,
				});
				if (result.status === 401) {
					logout();
					return;
				}
				if (result.json && result.json.room_ids) {
					state.rooms = result.json.room_ids;
					state.selectedRooms.forEach((roomId) => {
						if (!state.rooms.includes(roomId)) state.selectedRooms.delete(roomId);
					});
					if (!state.rooms.includes(state.activeRoom)) {
						state.activeRoom = "";
					}
					if (!state.activeRoom && state.rooms.length) {
						state.activeRoom = state.rooms[0];
					}
					clearInviteToken();
					await loadRoomMeta();
					if (!isAuthenticated()) return;
					updateActiveRoomLabel();
					updateInviteControls();
					renderRooms();
					renderSelectedRooms();
					await loadActiveHistory();
					setStatus("rooms loaded");
				} else {
					setStatus("failed to load rooms");
				}
			}

			async function loadRoomMeta() {
				if (!requireAuth()) return;
				const roomIds = state.rooms.slice();
				if (!roomIds.length) {
					state.roomMeta = {};
					return;
				}
				setStatus("loading room names...");
				const results = await Promise.all(
					roomIds.map((roomId) =>
						apiRequest("GET", "/room/info", {
							room_id: roomId,
							token: state.token,
						})
					)
				);
				const hasUnauthorized = results.some((result) => result.status === 401);
				if (hasUnauthorized) {
					logout();
					return;
				}
				const meta = {};
				roomIds.forEach((roomId, index) => {
					const result = results[index];
					if (result.json && result.json.room_id) {
						meta[roomId] = result.json;
					}
				});
				state.roomMeta = meta;
			}

				async function loadActiveHistory() {
					if (!requireAuth()) return;
					if (!state.activeRoom) {
						state.activeRoomMessages = [];
						closeRoomSocket();
						renderHistory([]);
						return;
					}
					updateActiveRoomLabel();
					setStatus("loading room messages...");
					const result = await apiRequest("GET", "/room/history", {
						room_id: state.activeRoom,
						token: state.token,
					});
					if (result.status === 401) {
						logout();
						return;
					}
					if (result.json && result.json.messages) {
						state.activeRoomMessages = result.json.messages;
						renderHistory(state.activeRoomMessages);
						connectRoomSocket();
						setStatus("room messages loaded (live)");
					} else {
						state.activeRoomMessages = [];
						closeRoomSocket();
						renderHistory([]);
						setStatus("failed to load room messages");
					}
				}

			async function login() {
				const uid = el("auth-uid").value.trim();
				const pwd = el("auth-pwd").value.trim();
				if (!uid || !pwd) {
					setAuthStatus("uid and pwd required");
					return;
				}
				setAuthStatus("logging in...");
				const result = await apiRequest("POST", "/account/login", { uid, pwd });
					if (result.json && result.json.token) {
						state.uid = uid;
						state.token = result.json.token;
						saveStored();
						el("auth-pwd").value = "";
						setAuthOverlayVisible(false);
						await refreshRooms();
						connectAgentStream();
						setStatus("logged in");
						setAuthStatus("logged in");
					} else {
						setAuthStatus("login failed");
					}
				}

			async function register() {
				const uid = el("auth-uid").value.trim();
				const pwd = el("auth-pwd").value.trim();
				if (!pwd) {
					setAuthStatus("pwd required");
					return;
				}
				setAuthStatus("registering...");
				const result = await apiRequest("POST", "/account/register", { uid, pwd });
				if (result.json && result.json.uid) {
					state.uid = result.json.uid;
					saveStored();
					el("auth-uid").value = result.json.uid;
					setAuthStatus("registered (now login)");
					setStatus("registered");
				} else {
					setAuthStatus("register failed");
				}
			}

					function logout() {
						setSettingsOverlayVisible(false);
						closeRoomSocket();
						closeAgentStream();
						state.token = "";
						saveStored();
						state.rooms = [];
						state.selectedRooms = new Set();
						state.agentThread = [];
						state.toolCalls = [];
						state.activeAgentRunId = "";
						state.activeRoom = "";
						state.activeRoomMessages = [];
						state.roomMeta = {};
						state.roomSocketRetryMs = 0;
						state.agentStreamRetryMs = 0;
						renderRooms();
						renderSelectedRooms();
						renderAgentThread();
						renderToolCalls();
						setToolCallsStatus("disconnected");
					clearInviteToken();
					updateInviteControls();
					el("active-room-label").textContent = "No room selected";
					renderHistory([]);
				setAuthOverlayVisible(true);
				setAuthStatus("logged out");
				setStatus("logged out");
			}

			async function createRoom() {
				if (!requireAuth()) return;
				const name = el("room-name").value.trim();
				const visibility = el("room-visibility").value.trim();
				if (!name) {
					setStatus("room name required");
					return;
				}
				const result = await apiRequest("POST", "/room/create", {
					uid: state.uid,
					name,
					visibility: visibility || "",
					token: state.token,
				});
				if (result.status === 401) {
					logout();
					return;
				}
				if (result.json && result.json.room_id) {
					await refreshRooms();
					setStatus("room created");
				} else {
					setStatus("room create failed");
				}
			}

			async function sendRoomMessage() {
				if (!requireAuth()) return;
				const msg = el("room-msg").value.trim();
				const roomId = state.activeRoom;
				if (!roomId || !msg) {
					setStatus("room and message required");
					return;
				}
				const result = await apiRequest("POST", "/room/talk", {
					uid: state.uid,
					room_id: roomId,
					msg,
					token: state.token,
				});
				if (result.status === 401) {
					logout();
					return;
					}
					if (result.json && result.json.message_id) {
						el("room-msg").value = "";
						if (!isLiveRoomConnected()) {
							await loadActiveHistory();
						}
						setStatus("message sent");
					} else {
						setStatus("message failed");
					}
			}

			async function createInvite() {
				if (!requireAuth()) return;
				if (!state.activeRoom) {
					setStatus("pick a room to invite");
					return;
				}
				clearInviteToken();
				setStatus("creating invite...");
				const result = await apiRequest("POST", "/room/invite", {
					uid: state.uid,
					room_id: state.activeRoom,
					token: state.token,
				});
				if (result.status === 401) {
					logout();
					return;
				}
				if (result.json && result.json.invite_token) {
					el("invite-token").value = result.json.invite_token;
					el("invite-box").hidden = false;
					setStatus("invite created");
				} else {
					setStatus("invite failed");
				}
			}

			async function copyInviteToken() {
				const token = (el("invite-token").value || "").trim();
				if (!token) return;
				try {
					if (navigator.clipboard && navigator.clipboard.writeText) {
						await navigator.clipboard.writeText(token);
						setStatus("invite copied");
						return;
					}
				} catch (err) {
					// fall through to manual copy below
				}
				window.prompt("Copy invite token:", token);
			}

			async function joinRoomByInvite() {
				if (!requireAuth()) return;
				const inviteToken = (el("join-invite").value || "").trim();
				if (!inviteToken) {
					el("settings-status").textContent = "invite token required";
					return;
				}
				el("settings-status").textContent = "joining room...";
				const result = await apiRequest("POST", "/room/join", {
					uid: state.uid,
					invite_token: inviteToken,
					token: state.token,
				});
				if (result.status === 401) {
					logout();
					return;
				}
				if (result.json && result.json.room_id) {
					state.activeRoom = result.json.room_id;
					el("join-invite").value = "";
					el("settings-status").textContent = "joined";
					await refreshRooms();
				} else {
					el("settings-status").textContent = "join failed";
				}
			}

				async function sendAgent() {
					if (!requireAuth()) return;
					const msg = el("agent-msg").value.trim();
					if (!msg) {
						setStatus("agent message required");
						return;
					}
					const roomIds = Array.from(state.selectedRooms).join(",");
					if (!roomIds) {
						setStatus("drag rooms into agent to select context");
						return;
					}
					// New agent run: clear tool-call trace and ensure the stream is connected.
					state.activeAgentRunId = makeRunId();
					state.toolCalls = [];
					renderToolCalls();
					const toolPanel = el("tool-calls-panel");
					if (toolPanel) toolPanel.open = true;
					setToolCallsStatus(isAgentStreamConnected() ? "connected • running" : "connecting...");
					connectAgentStream();

					state.agentThread.push({ role: "user", text: msg });
					renderAgentThread();
					el("agent-msg").value = "";
					setStatus("asking agent...");
					const result = await apiRequest("POST", "/agent", {
						user_id: state.uid,
						message: msg,
						room_ids: roomIds,
						token: state.token,
						run_id: state.activeAgentRunId,
						rulebook_id: state.rulebookId,
						rulebook_text: String(state.rulebookId || "").toLowerCase() === "custom" ? state.rulebookText : "",
					});
				if (result.status === 401) {
					logout();
					return;
				}
				if (result.json && result.json.response) {
					state.agentThread.push({ role: "agent", text: formatAgentResponse(result.json.response) });
					renderAgentThread();
					setStatus("agent responded");
				} else {
					const errorDetail =
						(result.json && (result.json.error || result.json.message)) ||
						`agent request failed (HTTP ${result.status})`;
					state.agentThread.push({ role: "system", text: errorDetail });
					renderAgentThread();
					setStatus("agent failed");
				}
			}

			function addRoomToContext(roomIdRaw) {
				const roomId = validateRoomId(roomIdRaw);
				if (!roomId) return;
				if (!state.selectedRooms.has(roomId)) {
					state.selectedRooms.add(roomId);
				}
				renderSelectedRooms();
				renderRooms();
				if (!state.activeRoom) {
					setActiveRoom(roomId);
				}
			}

					function bindEvents() {
						el("auth-login").addEventListener("click", login);
						el("auth-register").addEventListener("click", register);
					el("auth-settings").addEventListener("click", () => setSettingsOverlayVisible(true));
					el("auth-pwd").addEventListener("keydown", (event) => {
						if (event.key === "Enter") login();
					});
					el("open-settings").addEventListener("click", () => setSettingsOverlayVisible(true));
					el("close-settings").addEventListener("click", () => setSettingsOverlayVisible(false));
					el("logout").addEventListener("click", () => {
						setSettingsOverlayVisible(false);
						logout();
					});
					el("refresh-rooms").addEventListener("click", refreshRooms);
					el("create-room").addEventListener("click", createRoom);
					el("send-room").addEventListener("click", sendRoomMessage);
					el("invite-room").addEventListener("click", createInvite);
						el("copy-invite").addEventListener("click", copyInviteToken);
						el("send-agent").addEventListener("click", sendAgent);
						const rulebookSelect = el("rulebook-id");
						if (rulebookSelect) {
							rulebookSelect.addEventListener("change", () => {
								state.rulebookId = rulebookSelect.value;
								saveRulebookStored();
								syncRulebookUi();
							});
						}
						const rulebookArea = el("rulebook-text");
						if (rulebookArea) {
							rulebookArea.addEventListener("input", () => {
								state.rulebookText = rulebookArea.value;
								saveRulebookStored();
							});
						}
						el("join-room").addEventListener("click", joinRoomByInvite);
						el("join-invite").addEventListener("keydown", (event) => {
							if (event.key === "Enter") joinRoomByInvite();
						});
						el("save-base").addEventListener("click", () => {
							state.baseUrl = el("base-url").value.trim() || getAutoBaseUrl();
							saveBaseOverride(state.baseUrl);
							el("settings-status").textContent = "base url saved";
							setStatus("base url set");
							state.roomSocketRetryMs = 0;
							state.agentStreamRetryMs = 0;
							closeRoomSocket();
							closeAgentStream();
							setToolCallsStatus("disconnected");
							if (isAuthenticated()) {
								refreshRooms();
								connectAgentStream();
							}
						});
						el("use-window-base").addEventListener("click", () => {
							state.baseUrl = getAutoBaseUrl();
							saveBaseOverride("");
							el("base-url").value = state.baseUrl;
							el("settings-status").textContent = "using window base url";
							setStatus("base url set");
							state.roomSocketRetryMs = 0;
							state.agentStreamRetryMs = 0;
							closeRoomSocket();
							closeAgentStream();
							setToolCallsStatus("disconnected");
							if (isAuthenticated()) {
								refreshRooms();
								connectAgentStream();
							}
						});

				const agentDrop = el("agent-drop-zone");
				agentDrop.addEventListener("dragover", (event) => {
					event.preventDefault();
					agentDrop.style.borderColor = "var(--accent)";
				});
				agentDrop.addEventListener("dragleave", () => {
					agentDrop.style.borderColor = "var(--line)";
				});
				agentDrop.addEventListener("drop", (event) => {
					event.preventDefault();
					const roomId =
						event.dataTransfer.getData("application/x-rotunda-room") ||
						event.dataTransfer.getData("text/plain");
					addRoomToContext((roomId || "").trim());
					agentDrop.style.borderColor = "var(--line)";
				});
			}

					function init() {
						loadStored();
						renderSelectedRooms();
						renderAgentThread();
						renderToolCalls();
						updateInviteControls();
						bindEvents();
						setAuthOverlayVisible(!isAuthenticated());
						if (state.token && state.uid) {
							refreshRooms();
							connectAgentStream();
						} else {
							setToolCallsStatus("disconnected");
						}
					}

			init();
