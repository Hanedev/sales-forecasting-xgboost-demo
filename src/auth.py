from __future__ import annotations

import hashlib
import hmac

import streamlit as st

# Public recruiter demo account:
# username: demo
# password: demo123
#
# Only the salted password hash is stored in source control.
DEMO_SALT = bytes.fromhex("89e10119b5167de1d2b6b30d618a480b")
DEMO_HASH = bytes.fromhex("87c14e8b63e4c0632c786e47e90b2df7aa4d3c06e399d55f7e58437d59fbf2d6")


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000,
    )


def _verify_password(
    password: str,
    salt: bytes,
    expected_hash: bytes,
) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def _get_admin_credentials():
    try:
        config = st.secrets["auth"]["admin"]

        return {
            "username": str(config["username"]),
            "salt": bytes.fromhex(str(config["salt_hex"])),
            "password_hash": bytes.fromhex(
                str(config["password_hash_hex"])
            ),
        }
    except Exception:
        return None


def authenticate(username: str, password: str):
    username = username.strip()

    if username == "demo" and _verify_password(
        password,
        DEMO_SALT,
        DEMO_HASH,
    ):
        return {
            "username": "demo",
            "role": "demo",
        }

    admin = _get_admin_credentials()

    if admin and username == admin["username"]:
        if _verify_password(
            password,
            admin["salt"],
            admin["password_hash"],
        ):
            return {
                "username": username,
                "role": "admin",
            }

    return None


def init_auth_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("role", None)


def login_form() -> bool:
    init_auth_state()

    if st.session_state["authenticated"]:
        return True

    st.title("Sales Forecasting — XGBoost Demo")
    st.caption(
        "Portfolio application with synthetic data and "
        "lightweight authentication."
    )

    st.info(
        "Demo account — username: `demo` · password: `demo123`"
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Sign in",
            type="primary",
        )

    if submitted:
        user = authenticate(username, password)

        if user:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user["username"]
            st.session_state["role"] = user["role"]
            st.rerun()

        st.error("Invalid username or password.")

    return False


def logout_button():
    if st.sidebar.button("Sign out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.rerun()


def current_role() -> str | None:
    return st.session_state.get("role")
