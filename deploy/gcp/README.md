# Deploy to a GCP L4 in India

One VM runs everything: Whisper large-v3 (speech to text), Qwen3.5 4B (replies) and Svara TTS 3B
(speech), behind Caddy for HTTPS and an admin login. It serves **one conversation at a time**; a second
caller is refused.

## Before you start

- A project with billing, an `NVIDIA_L4_GPUS` quota in the region, and `GPUS_ALL_REGIONS` of at least 1.
- A network tag whose firewall rule allows tcp:80 and tcp:443. `create_vm.sh` uses `voice-agent`; set
  `TAG` to change it. No other port needs to be open.
- A domain pointing at the VM. Without one, `<ip-with-dashes>.sslip.io` works.
- Cost: the VM bills while it runs. Stop it when idle. A stopped VM still bills for its disk and static
  IP, and its zone may be out of L4s when you start it again.

## Install

```bash
PROJECT=my-project ./deploy/gcp/create_vm.sh

# From a checkout. uv.lock is not in Git, so make sure it exists: uv lock
tar -czf /tmp/app.tgz --exclude .venv --exclude .git --exclude media .
gcloud compute scp --zone ZONE /tmp/app.tgz deploy/gcp/provision.sh s2s-svara-l4:
gcloud compute ssh --zone ZONE s2s-svara-l4
```

On the VM:

```bash
sudo bash provision.sh voice.example.com ~/app.tgz
sudo /opt/s2s/app/deploy/gcp/update_admin.sh admin@example.com   # asks for the password
sudo systemctl start s2s
```

Run `provision.sh` again with a new archive to update.

## Operate

```bash
sudo systemctl status s2s          # also: start, stop, restart
sudo journalctl -fu s2s
sudo tail -f /var/lib/s2s/run/pipeline.log
gcloud compute instances stop s2s-svara-l4 --zone ZONE
```

To change the password, run `update_admin.sh` again and restart `s2s`. This also signs everyone out.

## Settings

`/etc/s2s/stack.env` holds the `S2S_*` settings, and `provision.sh` rewrites it. `/etc/s2s/admin.env`
holds the login, readable only by root.

| Setting | Effect |
|---|---|
| `S2S_WHISPER_MODEL=openai/whisper-large-v3-turbo` | Faster replies, but it mistakes several Indian languages for others. Download it first: downloads are off at run time |
| `S2S_WHISPER_LANGUAGES` | Languages Whisper may detect. Fewer is more reliable: `hi,en` if that is all you need |
| `S2S_SVARA_GENDER` | Default voice: `female` or `male` |

## How it is protected

Only Caddy listens publicly. The page and `/v1/realtime` need a signed session cookie (12 hours). Sign-in
is limited to 5 failures per address per 15 minutes. Requests from other sites are refused. The Realtime
server's other routes are not exposed.
