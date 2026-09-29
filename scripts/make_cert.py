"""Create a self-signed TLS certificate for LAN access (browsers require HTTPS for the microphone).

Writes certs/avatar.pem + certs/avatar-key.pem (git-ignored), valid for localhost, the PC name and
its LAN IP addresses. The browser shows a warning once; accept it for this address.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import socket
from pathlib import Path

import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

OUT = Path(__file__).resolve().parents[1] / "certs"


def lan_ips() -> list[str]:
    ips = []
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                ips.append(a.address)
    return ips


def main() -> None:
    OUT.mkdir(exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname()
    ips = lan_ips()
    names = [x509.DNSName("localhost"), x509.DNSName(host)] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ["127.0.0.1", *ips]]
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"AI Avatar ({host})")])
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .sign(key, hashes.SHA256()))
    (OUT / "avatar.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (OUT / "avatar-key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                           serialization.NoEncryption()))
    print(f"certificate for localhost, {host}, {', '.join(ips)} → {OUT}")
    for ip in ips:
        print(f"  open https://{ip}:8000/  (accept the warning once)")


if __name__ == "__main__":
    main()
