output "bootstrap_ips" {
  description = "Public IPs of bootstrap nodes by region"
  value = {
    for k, v in digitalocean_droplet.bootstrap : k => v.ipv4_address
  }
}

output "bootstrap_urls" {
  description = "Bootstrap URLs to put in openhydra_defaults.py"
  value = {
    for k, v in var.regions : k => "https://${v.hostname}"
  }
}

output "ssh_commands" {
  description = "SSH commands to connect to bootstrap nodes"
  value = {
    for k, v in digitalocean_droplet.bootstrap : k => "ssh root@${v.ipv4_address}"
  }
}
