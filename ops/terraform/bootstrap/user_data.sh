#!/bin/bash
set -euo pipefail

# Update and install deps
apt-get update -qq
apt-get install -y python3 python3-pip python3-venv curl ufw

# Install OpenHydra
if [ "${openhydra_version}" = "latest" ]; then
  pip3 install --break-system-packages openhydra
else
  pip3 install --break-system-packages "openhydra==${openhydra_version}"
fi

# Create systemd service
cat > /etc/systemd/system/openhydra-dht.service << 'SYSTEMD_EOF'
[Unit]
Description=OpenHydra DHT Bootstrap Node
After=network.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/openhydra-dht --host 0.0.0.0 --port ${dht_port} --ttl-seconds 300
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=openhydra-dht

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

# Enable firewall
ufw allow 22/tcp
ufw allow ${dht_port}/tcp
ufw --force enable

# Start DHT service
systemctl daemon-reload
systemctl enable openhydra-dht
systemctl start openhydra-dht
