#!/usr/bin/env bash
# Set the admin login. Run on the VM: sudo deploy/gcp/update_admin.sh admin@example.com
# The password is read from the terminal or stdin, and only its hash is stored.

set -euo pipefail

EMAIL="${1:?usage: update_admin.sh EMAIL}"
HASH="$(/opt/s2s/app/.venv/bin/python /opt/s2s/app/demo/admin_auth.py hash-password)"
umask 077
mkdir -p /etc/s2s
{
    echo "ADMIN_EMAIL=$EMAIL"
    echo "ADMIN_PASSWORD_HASH=$HASH"
    echo "ADMIN_SESSION_SECRET=$(openssl rand -hex 32)"
} > /etc/s2s/admin.env.new
mv /etc/s2s/admin.env.new /etc/s2s/admin.env
echo "Admin login saved. Apply it with: sudo systemctl restart s2s"
