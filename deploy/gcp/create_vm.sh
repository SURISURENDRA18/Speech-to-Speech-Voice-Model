#!/usr/bin/env bash
# Create an L4 VM in Mumbai. L4 stock varies, so every zone in the region is tried.
# The VM bills while it runs, whether or not anyone is talking to it.

set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-south1}"
NAME="${NAME:-s2s-svara-l4}"
MACHINE="${MACHINE:-g2-standard-8}"
# A network tag that has a firewall rule for tcp:80 and tcp:443.
TAG="${TAG:-voice-agent}"

gcloud compute addresses describe "$NAME-ip" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1 ||
    gcloud compute addresses create "$NAME-ip" --region "$REGION" --project "$PROJECT"
ADDRESS="$(gcloud compute addresses describe "$NAME-ip" --region "$REGION" --project "$PROJECT" --format='value(address)')"

for zone in "$REGION-a" "$REGION-b" "$REGION-c"; do
    echo "Trying $zone..."
    # Ubuntu 24.04 for glibc 2.39, driver 580 for CUDA 13. GPU VMs cannot live-migrate.
    if gcloud compute instances create "$NAME" \
        --project "$PROJECT" \
        --zone "$zone" \
        --machine-type "$MACHINE" \
        --maintenance-policy TERMINATE \
        --restart-on-failure \
        --image-family ubuntu-accelerator-2404-amd64-with-nvidia-580 \
        --image-project ubuntu-os-accelerator-images \
        --boot-disk-size 100GB \
        --boot-disk-type pd-balanced \
        --address "$ADDRESS" \
        --tags "$TAG" \
        --no-service-account \
        --no-scopes \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels app=speech-to-speech; then
        echo "Created $NAME in $zone at $ADDRESS"
        exit 0
    fi
done
echo "No L4 available in $REGION. Try again later or set REGION." >&2
exit 1
