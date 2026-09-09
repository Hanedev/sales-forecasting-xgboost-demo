import argparse
import hashlib
import secrets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a salted PBKDF2 hash "
            "for a Streamlit admin password."
        )
    )
    parser.add_argument("password")
    args = parser.parse_args()

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        args.password.encode("utf-8"),
        salt,
        200_000,
    )

    print(f'salt_hex = "{salt.hex()}"')
    print(
        f'password_hash_hex = "{password_hash.hex()}"'
    )


if __name__ == "__main__":
    main()
