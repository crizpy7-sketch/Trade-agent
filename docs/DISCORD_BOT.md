# Talking to the swarm in Discord

There are two separate Discord integrations, and it is worth being clear about
which is which:

| | Webhook (`notify.py`) | Bot (`bot.py`) |
| --- | --- | --- |
| Direction | Swarm → you | You ↔ swarm |
| Setup | A webhook URL | A bot token + allowlist |
| Can it receive? | **No** | Yes |
| What it does | Pushes the daily report | Answers questions on demand |

Your existing "MarketSwarm connected" integration is the **webhook**. It cannot
read messages — that is not a configuration problem, it is what a webhook is.
To ask the swarm things you need a bot as well. Both can run at once and they
do not interfere.

## What you can ask

```
!status         is the swarm healthy, when did it last run
!today          what it found this morning, and what it rejected
!ticker NVDA    live quote plus what the swarm said about it today
!why SPY        why that symbol was rejected today
!calibration    the track record so far
!agents         per-agent health — which ones are degraded
!help           this list
```

`!ticker` labels its two halves separately on purpose. The quote is fetched at
the moment you ask; the verdict is from the morning's run. A fresh price sitting
next to a stale verdict reads as one coherent statement and is not one.

## What it cannot do

The bot is read-only by construction, not by policy. It maps onto the read API
and the quote provider, and nothing else is reachable from it:

- It cannot trigger a pre-market run
- It cannot publish anything
- It cannot trade, and there is no execution path in MarketSwarm at all
- It cannot read or print secrets

## Setup

### 1. Create the bot application

Only you can do this step.

1. Go to <https://discord.com/developers/applications> → **New Application**
2. Name it (e.g. `MarketSwarm`)
3. **Bot** in the sidebar → **Reset Token** → copy the token. Treat it like a
   password: anyone holding it can post as your bot.
4. On the same page, scroll to **Privileged Gateway Intents** and turn on
   **MESSAGE CONTENT INTENT**. Without it Discord returns your messages with the
   `content` field blank and the bot sees every command as empty.

### 2. Invite it to your server

**OAuth2 → URL Generator**:

- Scopes: `bot`
- Bot permissions: `View Channels`, `Send Messages`, `Read Message History`

Open the generated URL and add it to your server. Those three permissions are
all it needs — it does not need admin, and should not have it.

### 3. Get the two ids

In Discord: **User Settings → Advanced → Developer Mode** on. Then:

- **Channel id** — right-click the channel → *Copy Channel ID*
- **Your user id** — right-click your own name → *Copy User ID*

### 4. Configure

Add to `/etc/marketswarm/env` on the VPS:

```
MARKETSWARM_DISCORD_BOT_TOKEN=your-bot-token
MARKETSWARM_DISCORD_CHANNEL_ID=123456789012345678
MARKETSWARM_DISCORD_ALLOWED_USERS=your-user-id
```

`MARKETSWARM_DISCORD_ALLOWED_USERS` accepts several ids separated by commas or
spaces.

**The allowlist fails closed.** With it empty the bot refuses to answer anyone
and says so at startup rather than starting and answering everyone. A research
command spends provider rate limit and API quota, and a Discord channel is
something other people can be invited to.

### 5. Install and start

```sh
sudo cp deploy/marketswarm-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now marketswarm-bot
sudo systemctl status marketswarm-bot
```

Then type `!help` in the channel.

## If it does not answer

```sh
journalctl -u marketswarm-bot -n 50 --no-pager --full
```

| What you see | What it means |
| --- | --- |
| `not configured: ... ALLOWED_USERS is empty` | Step 4 — the allowlist |
| Bot online but silent on every command | MESSAGE CONTENT INTENT is off (step 1.4) |
| `Not authorised` in the channel | Your user id is not in the allowlist. The reply includes the id to add. |
| `discord GET ... -> 403` | The bot is not in the channel, or lacks *Read Message History* |
| Nothing at all in the log | Wrong channel id — it is listening somewhere else |

A command repeated within 5 seconds is dropped without a reply. That is the
per-user cooldown, not a fault.

## How it works, briefly

The bot polls `GET /channels/{id}/messages` every 3 seconds rather than holding
a gateway WebSocket. That needs no extra package, no inbound port and no TLS
certificate, and it recovers from a network blip by simply asking again. The
cost is a few seconds of latency, on answers that take longer than that to
compute.

On start it reads the channel once and discards what it finds, so a restart
never replays and re-answers a backlog of old commands.
