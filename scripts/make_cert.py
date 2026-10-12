"""Create a self-signed TLS certificate for LAN access (browsers require HTTPS for the microphone).

Writes certs/avatar.pem + certs/avatar-key.pem (git-ignored), valid for localhost, the PC name and
its LAN IP addresses. The browser shows a warning once; accept it for this address.

    .venv\\Scripts\\python scripts\\make_cert.py [--port 8000]    (scripts\\run.ps1 -Https calls it)

Apple's rules for TLS server certificates (Safari on macOS and iOS refuse others, even after "accept"):
the names in subjectAltName, extended key usage serverAuth, at most 825 days of validity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import socket
from pathlib import Path

import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

OUT = Path(__file__).resolve().parents[1] / "certs"


def lan_ips() -> list[str]:
    ips = []
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                ips.append(a.address)
    return ips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="only for the printed links")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname()
    ips = lan_ips()
    names = [x509.DNSName("localhost"), x509.DNSName(host)] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ["127.0.0.1", *ips]]
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"AI Avatar ({host})")])
    now = dt.datetime.now(dt.UTC)
    not_before = now - dt.timedelta(minutes=5)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(not_before)
            .not_valid_after(not_before + dt.timedelta(days=825))      # Apple: 825 days at most, counted from not_before
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            # a server certificate, not a CA (Firefox refuses a CA certificate used for a server)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    (OUT / "avatar.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (OUT / "avatar-key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                           serialization.NoEncryption()))
    # ASCII only: a Windows console (cp1252) cannot encode arrows when the output is piped
    print(f"certificate for localhost, {host}, {', '.join(ips)} -> {OUT}")
    for ip in ips:
        print(f"  open https://{ip}:{args.port}/  (accept the warning once)")


if __name__ == "__main__":
    main()
