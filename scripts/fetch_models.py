#!/usr/bin/env python3
# Description: Download the official, encrypted WADAS AI models on a node without a graphical
# session, with the same calls the WADAS application makes.

"""Download the official WADAS models on this machine.

The models are encrypted for the hardware of the machine that downloads them, so this must be
run on every node. Run it from a terminal on the node: the e-mail and the token are asked
here and are never written anywhere (never pass them as arguments or put them in a file).

The first run registers the machine as a node of your organization, which uses one of the
nodes your access request allows, and saves the node identifier in ~/.wadas_node_id: the
following runs reuse it and register nothing.

  python scripts/fetch_models.py --list                   what the server offers, nothing else
  python scripts/fetch_models.py                          the default models
  python scripts/fetch_models.py --models NAME [NAME ...] the named models
"""

import argparse
import getpass
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
SERVER_URL = "https://api.wadas.it:8443/"
NODE_ID_FILE = Path.home() / ".wadas_node_id"


def select_models(available, names=None):
    """The models to download: the ones named, or the defaults of the server."""
    if names:
        by_name = {model["name"]: model for model in available}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Not offered by the server: {', '.join(missing)}")
        return [by_name[name] for name in names]
    return [model for model in available if model.get("is_default")]


def target_dir(model_dir, model):
    """Where a model is unpacked, as the WADAS application does."""
    return model_dir / model["path"].lstrip("/")


def ensure_real_model_dir(model_dir, unlink_link=False):
    """The models are written into `model_dir`: it must not be a link to another folder.

    A link would send the download into the folder it points to. It is only ever unlinked,
    never emptied or removed with its content.
    """
    if model_dir.is_symlink():
        if not unlink_link:
            raise SystemExit(
                f"{model_dir} is a link to {os.readlink(model_dir)}: the download would write "
                "there. Run again with --unlink-model-link to replace the link with a new "
                "folder (the folder it pointed to is left untouched)."
            )
        model_dir.unlink()
    model_dir.mkdir(exist_ok=True)


def read_node_id(path):
    return path.read_text().strip() if path.is_file() else None


def save_node_id(node_id, path):
    path.write_text(str(node_id) + "\n")
    path.chmod(0o600)


def describe(model):
    flag = "  (default)" if model.get("is_default") else ""
    return f"  {model['type']:15s} {model['name']}{flag}"


def get_node_id(server, assume_yes):
    node_id = read_node_id(NODE_ID_FILE)
    if node_id:
        print(f"Node already registered ({NODE_ID_FILE})")
        return node_id
    print(
        "This machine is not registered yet. Registering uses one of the nodes allowed by "
        "your access request, and cannot be undone from here."
    )
    if not assume_yes and input("Register this machine? [y/N] ").strip().lower() != "y":
        raise SystemExit("Not registered")
    email = input("E-mail: ").strip()
    token = getpass.getpass("Token: ")
    org_code = server.login(username=email, password=token)
    node_id = str(server.register_node(org_code=org_code))
    # Shown before it is saved: if saving failed, this identifier is the only trace of a
    # registration that cannot be repeated (put it in the file by hand to reuse it)
    print(f"Node identifier: {node_id}")
    save_node_id(node_id, NODE_ID_FILE)
    print(f"Registered: node saved in {NODE_ID_FILE}")
    return node_id


def download(server, node_id, models):
    for model in models:
        destination = target_dir(MODEL_DIR, model)
        archive = Path(f"{destination}.zip")
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model['name']} ...", flush=True)
        if not server.download_model(
            user_id=node_id, model_name=model["name"], model_path=str(archive), timeout=300
        ):
            print(f"  failed: {model['name']}")
            destination.rmdir()
            continue
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)
        archive.unlink()
        print(f"  done: {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="Only list the models on offer")
    parser.add_argument("--models", nargs="+", help="Names of the models to download")
    parser.add_argument("--unlink-model-link", action="store_true", help="Replace a model link")
    parser.add_argument("--yes", action="store_true", help="Do not ask before registering")
    args = parser.parse_args(argv)

    from wadas_runtime import WADASModelServer

    server = WADASModelServer(SERVER_URL)
    node_id = get_node_id(server, args.yes)
    available = server.available_models(node_id)
    print("Models on offer:")
    for model in available:
        print(describe(model))
    if args.list:
        return 0

    try:
        models = select_models(available, args.models)
    except ValueError as exc:
        print(exc)
        return 1
    if not models:
        print("Nothing to download: no default model, name them with --models")
        return 1
    ensure_real_model_dir(MODEL_DIR, args.unlink_model_link)
    download(server, node_id, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
