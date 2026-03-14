variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "regions" {
  description = "DigitalOcean regions for bootstrap nodes"
  type = map(object({
    slug     = string
    hostname = string
  }))
  default = {
    us = { slug = "nyc3", hostname = "bootstrap-us.openhydra.co" }
    eu = { slug = "ams3", hostname = "bootstrap-eu.openhydra.co" }
    ap = { slug = "sgp1", hostname = "bootstrap-ap.openhydra.co" }
  }
}

variable "droplet_size" {
  description = "Droplet size (s-1vcpu-1gb = $6/mo Nanode)"
  type        = string
  default     = "s-1vcpu-1gb"
}

variable "ssh_key_fingerprint" {
  description = "SSH public key fingerprint from DigitalOcean account"
  type        = string
}

variable "openhydra_version" {
  description = "OpenHydra package version or git ref"
  type        = string
  default     = "latest"
}

variable "dht_port" {
  description = "Port the DHT bootstrap listens on"
  type        = number
  default     = 8468
}
