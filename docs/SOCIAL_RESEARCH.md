# Permissioned community research

MarketSwarm can use community discussion as low-weight context without scraping
sites or automating a personal Discord account.

## Supported paths

1. **Discord bot channels.** Add the MarketSwarm bot to a server/channel where
   the owner permits it, grant only `View Channel` and `Read Message History`,
   then add the channel id to `MARKETSWARM_SOCIAL_DISCORD_CHANNEL_IDS`.
2. **TradingView alerts.** Create a private Discord intake channel and a Discord
   webhook for it. Put the webhook URL in the TradingView alert and format the
   alert as JSON, for example:

   ```json
   {"content":"$NVDA direction=long TradingView alert: breakout above 125"}
   ```

   You can also paste a public TradingView idea link into that intake channel.
   Include the ticker and direction (for example, `$NVDA direction=long`) unless
   Discord's preview already contains both. MarketSwarm reads only what Discord
   delivered; it never opens or crawls TradingView pages.
3. **Free Discord communities.** If a server has an announcement channel, use
   Discord's **Follow** feature to mirror permitted announcements into your own
   intake channel. Otherwise the community must allow your bot into the source
   channel, or you must forward posts manually. Never use a personal user token.
4. **X.** Set `MARKETSWARM_X_HANDLES` to an explicit allowlist and
   `X_BEARER_TOKEN` to an official X API bearer token. X API access is currently
   pay-per-use. The free alternative is to forward selected X post links into
   the Discord intake channel.

## What counts as evidence

- A post must mention a symbol in the configured universe and contain an
  explicit direction. Structured `direction=long` / `direction=short` is best.
- Repetition by one account counts once. Matching handles on different
  platforms are conservatively merged. By default, at least two accounts must
  agree and two-thirds of sources must point the same way. Different aliases
  used by the same person cannot be identified automatically, so this is an
  account-level check, not proof of independent humans.
- The cold-start probability nudge is 53/47 and is capped at 58/42. It cannot
  override price, volatility, option-chain, SEC, or review-gate evidence.
- Likes, followers, reposts, and Discord role names do not establish skill.
  MarketSwarm does not call anyone a "best trader" from popularity. Until an
  author-scoring layer is added and validated on resolved, time-stamped calls,
  every configured account stays at the same low cold-start weight.

## Configuration

Add these to `/etc/marketswarm/env`:

```sh
MARKETSWARM_SOCIAL_DISCORD_CHANNEL_IDS=123456789012345678
MARKETSWARM_COMMUNITY_LOOKBACK_HOURS=24
MARKETSWARM_COMMUNITY_MIN_SOURCES=2

# Optional official X API
MARKETSWARM_X_HANDLES=handle1,handle2
X_BEARER_TOKEN=replace_with_bearer_token
```

Then update and verify:

```sh
sudo bash deploy/update.sh
sudo marketswarm-cli status
sudo systemctl restart marketswarm marketswarm-bot
```

The morning Discord summary and `!plays` show exactly three call and three put
screening slots. Only `QUALIFIED` means the candidate survived both gates.
