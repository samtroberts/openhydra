# OpenHydra Bootstrap Nodes — Terraform

Provisions three geographically distributed DHT bootstrap nodes on DigitalOcean
(nyc3 / ams3 / sgp1) using the cheapest available Droplet size (~$6/mo each,
~$18/mo total for the full production fleet).

## Directory layout

```
ops/terraform/
├── bootstrap/              # Reusable module — one Droplet per region
│   ├── main.tf             # Droplets, firewall, DNS records
│   ├── variables.tf        # All input variables with defaults
│   ├── outputs.tf          # IPs, URLs, SSH commands
│   ├── versions.tf         # Terraform + provider version pins
│   └── user_data.sh        # Cloud-init: installs & starts openhydra-dht
└── environments/
    ├── production.tfvars   # Full three-region production deployment
    └── staging.tfvars      # Single nyc3 node for pre-prod testing
```

## Prerequisites

1. **Terraform >= 1.6** — https://developer.hashicorp.com/terraform/install
2. **DigitalOcean account** — https://cloud.digitalocean.com/
3. **SSH key uploaded to DigitalOcean** — Settings > Security > SSH Keys.
   Note the key fingerprint shown in the UI (format: `aa:bb:cc:...`).
4. **Domain delegated to DigitalOcean DNS** — Networking > Domains.
   The module creates `A` records under `openhydra.co`; that domain must
   be managed by DO for the `digitalocean_record` resource to work.
   If the domain is elsewhere, set `create_dns_records = false` and point your
   DNS manually to the IPs from `terraform output bootstrap_ips`.

## Getting a DigitalOcean API token

1. Log in at https://cloud.digitalocean.com/
2. Go to **API** > **Tokens** > **Generate New Token**
3. Grant **Read** and **Write** scopes
4. Copy the token — it is only shown once

## Deployment — production

```bash
cd ops/terraform/bootstrap

# Export secrets — never commit these
export TF_VAR_do_token="dop_v1_xxxxxxxxxxxxxxxxxxxx"
export TF_VAR_ssh_key_fingerprint="aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99"

terraform init
terraform plan -var-file=../environments/production.tfvars
terraform apply -var-file=../environments/production.tfvars
```

Terraform will create:

| Resource | Count |
|---|---|
| `digitalocean_droplet.bootstrap` | 3 (us, eu, ap) |
| `digitalocean_firewall.bootstrap` | 1 (shared) |
| `digitalocean_record.bootstrap` | 3 A records |

After `apply` completes, note the outputs:

```bash
terraform output bootstrap_ips       # raw IPs per region
terraform output bootstrap_urls      # full https:// URLs
terraform output ssh_commands        # ready-to-paste SSH commands
```

## Deployment — staging

Staging deploys a single node in nyc3 under a separate hostname so you can
validate the cloud-init script and systemd service without touching production
DNS.

```bash
cd ops/terraform/bootstrap

export TF_VAR_do_token="dop_v1_xxxxxxxxxxxxxxxxxxxx"
export TF_VAR_ssh_key_fingerprint="aa:bb:cc:..."

terraform init
terraform workspace new staging   # keeps state separate from production
terraform plan  -var-file=../environments/staging.tfvars
terraform apply -var-file=../environments/staging.tfvars
```

To switch back to the default (production) workspace:

```bash
terraform workspace select default
```

## Updating DNS after apply (external DNS)

If `openhydra.co` is not managed by DigitalOcean, remove or comment out
the `digitalocean_record` resource block in `main.tf`, then:

1. Run `terraform output bootstrap_ips` to get the assigned IPs.
2. In your DNS provider's control panel, create or update the following `A`
   records (TTL 300 is recommended):
   - `bootstrap-us.openhydra.co` → IP for region `us`
   - `bootstrap-eu.openhydra.co` → IP for region `eu`
   - `bootstrap-ap.openhydra.co` → IP for region `ap`

## Updating openhydra_defaults.py

The URLs in `openhydra_defaults.py` already match the hostnames in
`variables.tf`. No changes are needed unless you add or rename regions. If you
do rename a hostname:

1. Update the `hostname` field for that region in `variables.tf` (or in your
   `.tfvars` file).
2. Update the corresponding entry in `PRODUCTION_BOOTSTRAP_URLS` in
   `openhydra_defaults.py`.
3. Re-run `terraform apply` to update the DNS record.

## Verifying the service

After the Droplet boots (allow ~60 seconds for cloud-init):

```bash
# SSH in (get command from terraform output)
ssh root@<ip>

# Check service status
systemctl status openhydra-dht

# Tail live logs
journalctl -fu openhydra-dht

# Confirm port is listening
ss -tlnp | grep 8468
```

## Destroying infrastructure

```bash
cd ops/terraform/bootstrap

# Preview what will be deleted
terraform plan -destroy -var-file=../environments/production.tfvars

# Destroy (irreversible — Droplets, firewall, and DNS records are deleted)
terraform destroy -var-file=../environments/production.tfvars
```

For staging:

```bash
terraform workspace select staging
terraform destroy -var-file=../environments/staging.tfvars
terraform workspace select default
terraform workspace delete staging
```

## Cost estimate

| Component | Unit price | Count | Monthly |
|---|---|---|---|
| s-1vcpu-1gb Droplet (nyc3) | $6 | 1 | $6 |
| s-1vcpu-1gb Droplet (ams3) | $6 | 1 | $6 |
| s-1vcpu-1gb Droplet (sgp1) | $6 | 1 | $6 |
| Firewall | Free | 1 | $0 |
| DNS records | Free | 3 | $0 |
| **Total** | | | **~$18/mo** |

Staging (single node): ~$6/mo when running.

## Variables reference

| Variable | Default | Description |
|---|---|---|
| `do_token` | (required) | DigitalOcean API token |
| `ssh_key_fingerprint` | (required) | SSH key fingerprint from DO account |
| `regions` | us/eu/ap | Map of region key → DO slug + hostname |
| `droplet_size` | `s-1vcpu-1gb` | Droplet size slug |
| `openhydra_version` | `latest` | PyPI version or `latest` |
| `dht_port` | `8468` | DHT bootstrap listen port |
