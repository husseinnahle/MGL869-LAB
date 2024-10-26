import json
import requests
import schedule
import time
import threading


#########################
# Variables obligatoires
#########################
OUTPUT_FILE = "commits.json"  # Extraire les commits dans ce fichier
GITHUB_PAT = "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"  # Genérez un token dans github et collez le ici


GITHUB_URL = "https://api.github.com/repos"
REPO = "hive"
OWNER = "apache"

HEADERS = {
	"Accept": "application/vnd.github+json",
	"Authorization": "token " + GITHUB_PAT
}


# TODO: Il faudra peut-être ajuster cette fonction pour extraire les commits selon un release
def extract_commits(page: int = 1, per_page: int = 100) -> list[dict]:
    """Extract a list of commit hashes.

    Args:
        page (int, optional): The page number of the results to fetch. Defaults to 1.
        per_page (int, optional): The number of results per page (max 100). Defaults to 100.

    Returns:
        list[dict]: List of commit hashes.
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits?per_page={per_page}&page={page}", headers=HEADERS)
        return [e["sha"] for e in json.loads(response.text)]
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return {}


def extract_commit_content(hash: str) -> dict:
    """Extract the content of a single commit.

    Args:
        hash (str): Commit reference.

    Returns:
        dict: Dictionnary containning:
            - "message": The message of the commit.
            - "files": List of modified Java filenames. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits/" + hash, headers=HEADERS)
        commit = json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return {}
    return {
        "message": commit["commit"]["message"],
        "files": [e["filename"] for e in commit["files"] if e["filename"].endswith(".java") and e["status"] == "modified"]
    }


def _print_commit_count():
    print("Pulled %d commits." % len(commits))


def _start_scheduler():
    schedule.every(10).seconds.do(_print_commit_count)
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)


page = 1
per_page = 100
commits = []

stop_event = threading.Event()
scheduler_thread = threading.Thread(target=_start_scheduler)
scheduler_thread.start()

# Extract commits
commit_hashes = extract_commits(page, per_page)
while commit_hashes:
    for hash in commit_hashes:
        commit_content = extract_commit_content(hash)
        if commit_content:
            commits.append(commit_content)
    page += 1
    commit_hashes = extract_commits(page, per_page)

# Save commits content
with open(OUTPUT_FILE, "w") as file:
	json.dump(commits, file, indent=4)

stop_event.set()
scheduler_thread.join()
