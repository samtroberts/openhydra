provider "digitalocean" {
  token = var.do_token
}

# One Nanode per region
resource "digitalocean_droplet" "bootstrap" {
  for_each = var.regions

  name      = "openhydra-bootstrap-${each.key}"
  region    = each.value.slug
  size      = var.droplet_size
  image     = "ubuntu-24-04-x64"
  ssh_keys  = [var.ssh_key_fingerprint]
  user_data = templatefile("${path.module}/user_data.sh", {
    dht_port          = var.dht_port
    openhydra_version = var.openhydra_version
  })

  tags = ["openhydra", "bootstrap", each.key]
}

# Firewall: allow DHT port + SSH, deny everything else inbound
resource "digitalocean_firewall" "bootstrap" {
  name        = "openhydra-bootstrap"
  droplet_ids = [for d in digitalocean_droplet.bootstrap : d.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = tostring(var.dht_port)
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# DNS records (requires DigitalOcean managing the domain)
resource "digitalocean_record" "bootstrap" {
  for_each = var.regions

  domain = "openhydra.co"
  type   = "A"
  name   = "bootstrap-${each.key}"
  value  = digitalocean_droplet.bootstrap[each.key].ipv4_address
  ttl    = 300
}
